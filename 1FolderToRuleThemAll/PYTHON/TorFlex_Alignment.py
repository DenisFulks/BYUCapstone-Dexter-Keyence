import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
from copy import deepcopy
from functools import reduce
import time
from scipy.linalg import eigh

class Axle_Hub_LJS640:  
    def __init__(self, filename, view_angle_horizontal=0.0, scanType='real', cutOff=[-500, 500], ui=None):
        self.ui = ui
        self.minPoints = 10000
        self.anglePoints = 2000
        self.norm_tolerance_deg = 10.0
        self.dist_tolerange_mm = 15.0
        self.max_x_width = 640
        self.min_x_width = 579
        self.reference_z_depth = 1116
        self.min_z_depth = 936
        self.exp_norm = Normal_of_Rotated_Plane(axis='y', angle=view_angle_horizontal)
        
        if scanType == 'real' or scanType == 'live':
            data = np.loadtxt(filename, delimiter=',', dtype=np.float64)
            if scanType == 'real':
                z_scaling_factor = 1
            elif scanType == 'live':
                z_scaling_factor = 1
           
            self.numProfiles, self.numPoints = data.shape
            profile_indices = np.arange(self.numProfiles) - self.numProfiles / 2
            point_indices = (np.arange(self.numPoints) - self.numPoints / 2) * 0.2
            x, y = np.meshgrid(profile_indices, point_indices, indexing='ij')
            x = x.ravel()
            y = y.ravel()
            z = data.ravel() * z_scaling_factor
            valid_mask = (z >= -500) & (z <= 500)
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask]

            xScaleSlope = ((self.min_x_width - self.max_x_width) / self.numPoints) / \
                          (self.reference_z_depth - self.min_z_depth)
            scale_factors = np.where(z <= 0, 0.2, 0.2 + xScaleSlope*z)
            x *= scale_factors

            valid_mask = (x > -1000) & (x < 1000) & (z >= cutOff[0]) & (z <= cutOff[1])
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask]

            self.cloud = np.array([y, -x, z])
            self.numPoints = self.cloud.shape[1]
        elif scanType == 'sim':
            data = np.loadtxt(filename, skiprows=1)
            self.numProfiles, self.numPoints = data.shape
            x = data[:,0]
            y = data[:,1]
            z = data[:,2]
            self.cloud = np.array([x, y, z])
            self.numPoints = self.cloud.shape[1]
        
    def downsample_cloud(self, maxPoints):
        if self.numPoints > maxPoints:
            sampled_indices = np.linspace(0, self.numPoints - 1, maxPoints, dtype=int)
            self.cloud = self.cloud[:, sampled_indices]
            self.numPoints = self.cloud.shape[1]

    def trim_cloud_z(self, cutOff=[-500, 500]):
        valid_mask = (self.cloud[2] >= cutOff[0]) & (self.cloud[2] <= cutOff[1])
        myCloud = self.cloud[:, valid_mask]
        if myCloud.shape[1] >= self.minPoints:
            self.cloud = myCloud
            self.numPoints = self.cloud.shape[1]
        else:    
            if self.ui:
                self.ui.log_message(f'Trimming Error: Less than {self.minPoints} points in trimmed cloud. Returning untrimmed cloud')
            else:
                print(f'Trimming Error: Less than {self.minPoints} points in trimmed cloud. Returning untrimmed cloud')
        
    def show_cloud(self, altCloud=0):
        if isinstance(altCloud, int):
            Plot_Cloud_PyVista(self.cloud, pointSize=0.5, ui=self.ui)
        else:
            Plot_Cloud_PyVista(altCloud, pointSize=0.5, ui=self.ui)

    def align_z(self, auto=True):
        myCloud, R_matrix = Broadface_to_Z(self.cloud, self.exp_norm, self.norm_tolerance_deg, ui=self.ui)
        self.cloud = myCloud
        if auto == False:
            if self.ui:
                self.ui.log_message('Showing rotated cloud')
            else:
                print('Showing rotated cloud')
            for row in R_matrix:
                if self.ui:
                    self.ui.log_message(' '.join(f'{elem:.6f}' for elem in row))
                else:
                    print(' '.join(f'{elem:.6f}' for elem in row))
            self.show_cloud(myCloud)
            ans = input('Keep cloud after this rotation: y/n')
            if ans == 'n':
                if self.ui:
                    self.ui.log_message('Rejecting cloud. Terminating attempt')
                else:
                    print('Rejecting cloud. Terminating attempt')
                self.cloud = np.nan
        
    def rotate(self, axis, angle):
        angle = np.radians(angle)
        if axis == 'x':
            R = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])    
        elif axis == 'y':
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
        elif axis == 'z':
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")
        self.cloud = np.dot(R, np.array(self.cloud))

    def sort_ledges(self):
        self.cloud = Cloud_Expected_Normal_Filter(self.cloud, self.exp_norm, angle_threshold=self.norm_tolerance_deg, ui=self.ui)
        myLedges, myLedgeAvgs = Find_Ledges_Along_Normal(self.cloud, normal=[0, 0, 1], ledgeThreshold=2.0, shortLedge=0.01, closeLedges=4.5, ui=self.ui)
        self.sorted_ledges, self.sorted_ledge_avgs = Sort_Ledges(myLedges, myLedgeAvgs)

    def calc_ref_angle(self, index=0, plotNum=0):
        self.ref_ledge = self.sorted_ledges[index]
        self.ref_plane, self.ref_angle, _ = Calc_Plane(self.ref_ledge, numPoints=self.anglePoints, plotNum=plotNum, ui=self.ui)

    def calc_hub_angle(self, index=None, auto=True, plotNum=0, deleteGround=False):
        if index != None:
            self.hub_plane, self.hub_angle, _ = Calc_Plane(self.sorted_ledges[index], numPoints=self.anglePoints, plotNum=plotNum, ui=self.ui)
        else:
            self.hub_ledge, self.hub_avg = Find_HubFace(self.sorted_ledges, self.sorted_ledge_avgs, deleteGround=deleteGround, ui=self.ui)
            
            if auto == True:
                self.hub_plane, self.hub_angle, _ = Calc_Plane(self.hub_ledge, numPoints=self.anglePoints, plotNum=plotNum, ui=self.ui)
            
            elif auto == False:
                ans = 'No'
                while ans != 'Yes':
                    self.show_cloud(self.hub_ledge)
                    ans = self.ui.get_input(message='Is this the hub face?')
                    if ans == 'Yes':
                        break
                    if self.hub_avg in self.sorted_ledge_avgs:
                        current_index = self.sorted_ledge_avgs.index(self.hub_avg)
                    else:
                        if self.ui:
                            self.ui.log_message("Error: Hub average not found in sorted ledge averages.")
                        else:
                            print("Error: Hub average not found in sorted ledge averages.")
                        return
                    
                    ans2 = self.ui.get_input(message='Move to next ledge up or down?', options=["Up", "Down"])
                    while ans2 not in ('Up', 'Down'):
                        ans2 = input('Input not allowed. Move to ledge up or down: u/d ')
                    if ans2 == 'Up' and current_index < len(self.sorted_ledges) - 1:
                        current_index += 1
                    elif ans2 == 'Down' and current_index > 0:
                        current_index -= 1
                    else:
                        if self.ui:
                            self.ui.log_message("Cannot move further in that direction.")
                        else:
                            print("Cannot move further in that direction.")
                    self.hub_ledge = self.sorted_ledges[current_index]
                    self.hub_avg = self.sorted_ledge_avgs[current_index]
                self.hub_plane, self.hub_angle, _ = Calc_Plane(self.hub_ledge, numPoints=self.anglePoints, plotNum=plotNum, ui=self.ui)

    def calc_hub_relative_angle(self):
        self.hub_relative_angle = self.hub_angle - self.ref_angle

class Torsion_Arm_LJS640:
    def __init__(self, filename, view_angle_horizontal=0.0, scan_type='real', side='right', cutOff=[-500,500,-500,500,-500,500], ui=None):
        self.filename = filename
        self.ui = ui
        self.closeLedges = 0.1
        self.ledgeThreshold = 0.1
        self.barFaceRadius = 120
        self.minPoints = 10000
        self.anglePoints = 8000
        self.norm_tolerance_deg = 10.0
        self.dist_tolerange_mm = 15.0
        self.max_x_width = 640
        self.min_x_width = 579
        self.reference_z_depth = 1116
        self.min_z_depth = 936
        self.exp_norm = Normal_of_Rotated_Plane(axis='x', angle=view_angle_horizontal)
        self.side = side
        self.scan_type = scan_type
        
        if scan_type == 'real' or scan_type == 'live':
            data = np.loadtxt(filename, delimiter=',', dtype=np.float64)
            if scan_type == 'real':
                z_scaling_factor = 1
            elif scan_type == 'live':
                z_scaling_factor = 1
        
            self.numProfiles, self.numPoints = data.shape
            profile_indices = np.arange(self.numProfiles) - self.numProfiles / 2
            point_indices = (np.arange(self.numPoints) - self.numPoints / 2) * 0.2
            x, y = np.meshgrid(profile_indices, point_indices, indexing='ij')
            x = x.ravel()
            y = y.ravel()
            z = data.ravel() * z_scaling_factor
            valid_mask = (z >= -1000) & (z <= 1000) # Don't load points out of scanner range (empty space)
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask]

            xScaleSlope = ((self.min_x_width - self.max_x_width) / self.numPoints) / \
                        (self.reference_z_depth - self.min_z_depth)
            scale_factors = np.where(z <= 0, 0.2, 0.2 + xScaleSlope*z)
            x *= scale_factors

            valid_mask = (x > cutOff[0]) & (x < cutOff[1]) & (y > cutOff[2]) & (y < cutOff[3]) & (z >= cutOff[4]) & (z <= cutOff[5])
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask]

            self.cloud = np.array([x, y, z])
            self.numPoints = self.cloud.shape[1]
        
        elif scan_type == 'sim':
            data = np.loadtxt(filename, skiprows=1)
            self.numProfiles, self.numPoints = data.shape
            x = data[:,0]
            y = data[:,1]
            z = data[:,2]
            self.cloud = np.array([x, y, z])
            self.numPoints = self.cloud.shape[1]

    def downsample_cloud(self, maxPoints):
        '''Load fewer points for processing by evenly sampling over xy grid'''
        if self.numPoints > maxPoints:
            sampled_indices = np.linspace(0, self.numPoints - 1, maxPoints, dtype=int)
            self.cloud = self.cloud[:, sampled_indices]
            self.numPoints = self.cloud.shape[1]

    def center_cloud_xy(self):
        '''Positions scan in center of xy plane. Does not adjust z values'''
        centroid_xy = [np.mean(self.cloud[0]), np.mean(self.cloud[1])]
        self.cloud = self.cloud - np.array([centroid_xy[0], centroid_xy[1], 0])[:, np.newaxis]

    def rotate_cloud(self, axis, angle):
        '''Pure rotation of the cloud about the specified axis'''
        self.cloud = Rotate(self.cloud, axis, angle)
        
    def show_cloud(self, altCloud=0):
        '''Shows cloud in separate window using PyVista with z value color gradient. If no cloud if provided, shows raw scan'''
        if isinstance(altCloud, int):
            Plot_Cloud_PyVista(self.cloud, pointSize=0.5)
        else:
            Plot_Cloud_PyVista(altCloud, pointSize=0.5)

    def fit_bar_faces(self, cutoff=[-500, 500, -500, 500, -500, 500], plotNum=0, show=False, num_points=8000):
        '''Fits a plane to each of the bar surfaces'''
        barCloud = Trim_Cloud(self.cloud, 'x', [cutoff[0], cutoff[1]])
        # print('Showing bar cloud'); self.show_cloud(barCloud)
        barCloud = Trim_Cloud(barCloud, 'y', [cutoff[2], cutoff[3]])
        # print('Showing bar cloud'); self.show_cloud(barCloud)
        barCloud = Trim_Cloud(barCloud, 'z', [cutoff[4], cutoff[5]])   #70, 500
        # if show:
        #     print('Showing bar cloud'); self.show_cloud(barCloud)

        # Find primary face
        barPrimaryFaces = Cloud_Expected_Normal_Filter(barCloud, self.exp_norm, angle_threshold=6)  #6
        primaryLedges, primaryLedgeAvgs = Find_Ledges_Along_Normal(barPrimaryFaces, normal=self.exp_norm, ledgeThreshold=self.ledgeThreshold, shortLedge=0.01, closeLedges=self.closeLedges)
        self.barPrimaryFace = Sort_Ledges(primaryLedges, primaryLedgeAvgs, sortType='size')[0][-1]
        # self.show_cloud(self.barPrimaryFace)
        self.barPrimaryFace = Clean_Bar_Face(self.barPrimaryFace, radius=self.barFaceRadius)
        # self.show_cloud(self.barPrimaryFace)
        barPrimaryPlane, _, _ = Calc_Plane(self.barPrimaryFace, plotNum=plotNum, numPoints=num_points)
        barPrimaryNormal = barPrimaryPlane[0:3]

        # Find secondary face, which is perpendicular to primary
        exp_secondary_norm = Rotate(barPrimaryNormal, axis='x', angle=90.0)
        barSecondaryFaces = Cloud_Expected_Normal_Filter(barCloud, exp_secondary_norm, 3)   # 3
        secondaryLedges, secondaryLedgeAvgs = Find_Ledges_Along_Normal(barSecondaryFaces, normal=exp_secondary_norm, ledgeThreshold=self.ledgeThreshold, shortLedge=0.1, closeLedges=self.closeLedges)
        
        # for ledge in secondaryLedges:
        #     self.show_cloud(ledge)
        
        self.barSecondaryFace = Sort_Ledges(secondaryLedges, secondaryLedgeAvgs, sortType='size')[0][-1]
        # self.show_cloud(self.barSecondaryFace)
        self.barSecondaryFace = Clean_Bar_Face(self.barSecondaryFace, radius=self.barFaceRadius)
        # self.show_cloud(self.barSecondaryFace)
        barSecondaryPlane, _, _ = Calc_Plane(self.barSecondaryFace, plotNum=plotNum*2, numPoints=num_points)
        barSecondaryNormal = barSecondaryPlane[0:3]

        # Bar's axis is the intersection of primary and secondary faces
        self.bar_axis = np.cross(barPrimaryNormal, barSecondaryNormal)
        if self.bar_axis[0] < 0:
            self.bar_axis = -self.bar_axis
        self.bar_faces = np.hstack((self.barPrimaryFace, self.barSecondaryFace))
        highest_y_idx = np.argmax(self.bar_faces[1])
        self.bar_faces_highest_point = self.bar_faces[:, highest_y_idx]
        if show:
            print('Showing bar surfaces'); self.show_cloud(np.hstack((self.barPrimaryFace, self.barSecondaryFace)))

    def fit_spindle(self, axial_cutoff=-145, num_bins=20, circle_fit_tol=0.3, show=False, plot=False):
        '''
        Parameters:
            axial_cutoff: value above which all points are the spindle. Discards dogbone and bar below
            num_bins: number of segments to slice spindle into
            circle_fit_tol: upper (root mean square error) tolerance for accepting a slice's center 
                            based on how tightly the points fit a circle
            show: flag to display regions of scan while algorithm finds spindle and axis
            plot: flag to display each slice's projected circle fit
        '''
        min_points_per_bin=10
        '''Start of finding spindle within cloud'''
        # Set up frame based on bar axis
        approx_axis = self.bar_axis
        if abs(approx_axis[0]) < min(abs(approx_axis[1]), abs(approx_axis[2])):
            u = np.array([1, 0, 0])
        elif abs(approx_axis[1]) < abs(approx_axis[2]):
            u = np.array([0, 1, 0])
        else:
            u = np.array([0, 0, 1])
        u = u - np.dot(u, approx_axis) * approx_axis
        u = u / np.linalg.norm(u)
        v = np.cross(approx_axis, u)
        
        # Select half of scan which includes spindle
            # Calculate distances along axis from starting_point
        starting_point = self.bar_faces_highest_point
        delta = self.cloud.T - starting_point
        s = delta @ approx_axis
        mask = (s <= axial_cutoff)
        spindle_half = self.cloud.T[mask, :]
        if show:
            self.show_cloud(spindle_half.T)
        
        # Separate spindle from other objects
            # Project all points onto plane orthogonal to bar axis and find spindle
        plane_points = np.dot(spindle_half, np.array([u, v]).T)
        plane_spindle = Bound_Spindle_2D(plane_points, show=show)
        
        # Map planar spindle points back to original 3D cloud
        plane_points = np.dot(spindle_half, np.array([u, v]).T)  # Recalculate plane_points
            # Create a mask for points in plane_spindle
        mask = np.isin(plane_points, plane_spindle).all(axis=1)
        spindle_bounded = spindle_half[mask]
        if show:
            self.show_cloud(spindle_bounded.T)
        '''End of finding spindle within cloud'''
        ''''Start of fitting axis to spindle'''
        # Project spindle points onto bar axis and slice into bins
        t = np.dot(spindle_bounded, approx_axis)
        t_min, t_max = np.min(t), np.max(t)
        bin_edges = np.linspace(t_min, t_max, num_bins + 1)
        centers = []
        count = 0
        for i in range(num_bins):
            mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
            if np.sum(mask) < min_points_per_bin:
                continue
            points_bin = spindle_bounded[mask]
            
            # Project points in bin onto the plane orthogonal to bar axis
            points_2d = np.dot(points_bin, np.array([u, v]).T)
            maxC = np.max(points_2d)
            A = np.hstack([points_2d, np.ones((len(points_2d), 1))])
            b = -(points_2d[:, 0]**2 + points_2d[:, 1]**2)
            
            # Fit a circle to projected points and record its center in global 3D frame
            try:
                abc = np.linalg.lstsq(A, b, rcond=None)[0]
                a, b, c = abc
                discriminant = a**2 + b**2 - 4*c
                if discriminant > 0:
                    center_2d = [-a / 2, -b / 2]
                    radius = np.sqrt((a/2)**2 + (b/2)**2 - c)
                    residuals = np.sqrt((points_2d[:, 0] - center_2d[0])**2 + (points_2d[:, 1] - center_2d[1])**2) - radius
                    rmse = np.sqrt(np.mean(residuals**2))
                    if rmse < circle_fit_tol:
                        count += 1
                        t_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                        center_3d = t_center * approx_axis + center_2d[0] * u + center_2d[1] * v
                        centers.append(center_3d)
            except np.linalg.LinAlgError:
                continue
            if plot:# and i < 10:
                # self.show_cloud(points_bin.T)
                theta = np.linspace(0, 2 * np.pi, 100)
                x_circle = center_2d[0] + radius * np.cos(theta)
                y_circle = center_2d[1] + radius * np.sin(theta)
                plt.plot(x_circle, y_circle, 'r-', label='Best-fit circle', linewidth=0.5)
                plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1)
                plt.scatter(center_2d[0], center_2d[1])
                plt.title(f"Projected Slice {i}. rmse: {rmse}")
                plt.xlim(-maxC, maxC)
                plt.ylim(-maxC, maxC)
                plt.axis('equal')
                plt.xlabel("u-axis")
                plt.ylabel("v-axis")
                plt.show()
        
        # Fit a line to 3D centers and evaluate fit quality
        if len(centers) < 2:
            raise ValueError("Not enough valid circle fits to determine the axis.")
        centers = np.array(centers)
        np.savetxt(r'C:\Users\Public\CapstoneUI\centers.csv', centers, delimiter=',', header='X Y Z')
        print(f'Fitting axis to {len(centers)} of {num_bins} spindle slice centers')
        c_axis = np.mean(centers, axis=0)
        # PCA for line direction
        U, S, Vt = np.linalg.svd(centers - c_axis, full_matrices=False)
        axis_dir = Vt[0]  # Principal component (largest singular value)
        if np.dot(axis_dir, approx_axis) < 0:
            axis_dir = -axis_dir
        # Calculate fit quality (RMSE of perpendicular distances)
        projections = np.dot(centers - c_axis, axis_dir)
        points_on_line = c_axis + np.outer(projections, axis_dir)
        distances = np.linalg.norm(centers - points_on_line, axis=1)
        rmse = np.sqrt(np.mean(distances**2))
        print(f'Axis fit rmse: {rmse}')
        self.axis_loc = c_axis
        self.spindle_axis = axis_dir
        self.spindle_cloud = spindle_bounded.T
        self.line_fit_rmse = rmse  # Store fit quality
        if show:
            pcd = Numpy_to_Open3D(self.spindle_cloud)
            #visualize_axis(pcd, c_axis, axis_dir, length=100)

    def fit_spindle2(self, axial_cutoff=-145, num_bins=20, side='left', circle_fit_tol=0.3, circle_resid_tol=[1.0], min_fit_points=200, centers_resid_tol=[1.0], show=False, plot=False):
        '''
        Parameters:
            axial_cutoff: value above which all points are the spindle. Discards dogbone and bar below
            num_bins: number of segments to slice spindle into
            circle_fit_tol: upper (root mean square error) tolerance for accepting a slice's center 
                            based on how tightly the points fit a circle
            show: flag to display regions of scan while algorithm finds spindle and axis
            plot: flag to display each slice's projected circle fit
        '''
        min_points_per_bin=10
        '''Start of finding spindle within cloud'''
        # Set up frame based on bar axis

        approx_axis = self.bar_axis
        if abs(approx_axis[0]) < min(abs(approx_axis[1]), abs(approx_axis[2])):
            u = np.array([1, 0, 0])
        elif abs(approx_axis[1]) < abs(approx_axis[2]):
            u = np.array([0, 1, 0])
        else:
            u = np.array([0, 0, 1])
        u = u - np.dot(u, approx_axis) * approx_axis
        u = u / np.linalg.norm(u)
        v = np.cross(approx_axis, u)
        
        # Select half of scan which includes spindle
            # Calculate distances along axis from starting_point
        spindle_half = self.select_spindle_points(axial_cutoff, side)
        self.spindle_cloud = spindle_half
        # if show:
        #     self.show_cloud(spindle_half.T)
        
        # Separate spindle from other objects
            # Project all points onto plane orthogonal to bar axis and find spindle
        plane_points = np.dot(spindle_half, np.array([u, v]).T)
        plane_spindle = Bound_Spindle_2D(plane_points, show=show)
        
        # Map planar spindle points back to original 3D cloud
        plane_points = np.dot(spindle_half, np.array([u, v]).T)  # Recalculate plane_points
        # Create a mask for points in plane_spindle
        #mask = np.isin(plane_points, plane_spindle).all(axis=1)
        spindle_bounded = spindle_half
        if show:
            self.show_cloud(spindle_bounded.T)
        '''End of finding spindle within cloud'''
        ''''Start of fitting axis to spindle'''
        # Project spindle points onto bar axis and slice into bins
        t = np.dot(spindle_bounded, approx_axis)
        t_min, t_max = np.min(t), np.max(t)
        bin_edges = np.linspace(t_min, t_max, num_bins + 1)
        centers = []
        count = 0
        for i in range(num_bins):
            mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
            if np.sum(mask) < min_points_per_bin:
                continue
            points_bin = spindle_bounded[mask]
            
            def fit_circle(points_2d):
                maxC = np.max(points_2d)
                A = np.hstack([points_2d, np.ones((len(points_2d), 1))])
                b = -(points_2d[:, 0]**2 + points_2d[:, 1]**2)

                # Fit a circle to projected points and record its center in global 3D frame
                try:
                    abc = np.linalg.lstsq(A, b, rcond=None)[0]
                    a, b, c = abc
                    discriminant = a**2 + b**2 - 4*c
                    if discriminant > 0:
                        center_2d = [-a / 2, -b / 2]
                        radius = np.sqrt((a/2)**2 + (b/2)**2 - c)
                except np.linalg.LinAlgError:
                    return 0, 0, -1

                return center_2d, maxC, radius

            def filter_circle(points, center, iqr_scale):
                radii = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
                Q1, Q3 = np.percentile(radii, [25, 75])
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_scale * IQR
                upper_bound = Q3 + iqr_scale * IQR
                filtered_points = points[(radii >= lower_bound) & (radii <= upper_bound), :]
                return filtered_points

            filt_points = points_bin.T
            points_2d = np.dot(filt_points.T, np.array([u, v]).T)
            center_2d, maxC, radius = fit_circle(points_2d)
            for j, iqr_scale in enumerate(circle_resid_tol):
                if (radius < 0):
                    continue

                points_2d = filter_circle(points_2d, center_2d, iqr_scale)
                center_2d, maxC, radius = fit_circle(points_2d)

                residuals = np.sqrt((points_2d[:, 0] - center_2d[0])**2 + (points_2d[:, 1] - center_2d[1])**2) - radius
                rmse = np.sqrt(np.mean(residuals**2))

                if self.ui:
                    self.ui.log_message(f"\tSlice {i} Iteration {j}: filtering {points_2d.shape[0]} points, rmse: {rmse:.4f}")
                else:
                    print(f"\tSlice {i} Iteration {j}: filtering {points_2d.shape[0]} points, rmse: {rmse:.4f}")

                if plot and i % 10 == 0:
                    # self.show_cloud(points_bin.T)
                    theta = np.linspace(0, 2 * np.pi, 100)
                    x_circle = center_2d[0] + radius * np.cos(theta)
                    y_circle = center_2d[1] + radius * np.sin(theta)
                    plt.plot(x_circle, y_circle, 'r-', label='Best-fit circle', linewidth=0.5)
                    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1)
                    plt.scatter(center_2d[0], center_2d[1])
                    plt.title(f"Projected Slice {i}: Iteration: {j}, rmse: {rmse:.4f}, Points: {points_2d.shape[0]}")
                    plt.figtext(0.25, 0.0, f"IQR Scale: {iqr_scale}, Center = ({center_2d[0]}, {center_2d[1]})")
                    plt.xlim(-maxC, maxC)
                    plt.ylim(-maxC, maxC)
                    plt.axis('equal')
                    plt.xlabel("u-axis")
                    plt.ylabel("v-axis")
                    plt.show() 

            if points_2d.shape[0] >= min_fit_points:
                residuals = np.sqrt((points_2d[:, 0] - center_2d[0])**2 + (points_2d[:, 1] - center_2d[1])**2) - radius
                rmse = np.sqrt(np.mean(residuals**2))
                if rmse < circle_fit_tol:
                    count += 1
                    t_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                    center_3d = t_center * approx_axis + center_2d[0] * u + center_2d[1] * v
                    centers.append(center_3d)

        def filter_centers(centers, direction, iqr_scale, plot=False):
            if direction == "x":
                index = 0
            elif direction == "z":
                index = 2
            
            trend = np.polyfit(centers[:, 1], centers[:, index], 1)

            if plot:
                plt.clf()
                plt.title(f"Iteration: {i}, IQR Scale: {iqr_scale}")
                plt.scatter(centers[:, 1], centers[:, index])
                plt.plot([np.min(centers[:, 1]), np.max(centers[:, 1])], [trend[0] * np.min(centers[:, 1]) + trend[1], trend[0] * np.max(centers[:, 1]) + trend[1]], 
                         linestyle='dashed', color='r')
                plt.xticks(np.linspace(np.min(centers[:, 1]), np.max(centers[:, 1]), 7))
                plt.yticks(np.linspace(np.min(centers[:, index]), np.max(centers[:, index]), 5))
                plt.ylabel(f"{direction}-axis")
                plt.xlabel("y-axis")
                plt.show()

            distances = centers[:, index] - (trend[0] * centers[:, 1] + trend[1])

            Q1, Q3 = np.percentile(distances, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_scale * IQR
            upper_bound = Q3 + iqr_scale * IQR
            filtered_centers = centers[(distances >= lower_bound) & (distances <= upper_bound), :]
            return filtered_centers, trend
        
        if len(centers) < 2:
                raise ValueError("Not enough valid circle fits to determine the axis.")
        centers = np.array(centers)

        for i, iqr_scale in enumerate(centers_resid_tol):
            print(f'Iteration {i + 1}: Fitting axis to {len(centers)} of {num_bins} spindle slice centers')
            centers, trend_z = filter_centers(centers, "z", iqr_scale, plot=plot)
            centers, trend_x = filter_centers(centers, "x", iqr_scale, plot=plot)
        
        np.savetxt(r'C:\Users\Public\CapstoneUI\centersFiltered.csv', centers, delimiter=',', header='X Y Z')

        print(f'Filtered Fitting axis to {len(centers)} of {num_bins} spindle slice centers')
        c_axis = np.mean(centers, axis=0)
        # PCA for line direction
        U, S, Vt = np.linalg.svd(centers - c_axis, full_matrices=False)
        axis_dir = Vt[0]  # Principal component (largest singular value)
        if np.dot(axis_dir, approx_axis) < 0:
            axis_dir = -axis_dir
        # Calculate fit quality (RMSE of perpendicular distances)
        projections = np.dot(centers - c_axis, axis_dir)
        points_on_line = c_axis + np.outer(projections, axis_dir)
        distances = np.linalg.norm(centers - points_on_line, axis=1)
        rmse = np.sqrt(np.mean(distances**2))

        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=1)
            plt.setp(axes, xticks=[], yticks=[])

            plt.sca(axes[0])
            #plt.title(f"Iteration: {i}, rmse: {rmse:.4f}, IQR Scale: {iqr_scale}")
            plt.scatter(centers[:, 1], centers[:, 0])
            plt.plot([np.min(centers[:, 1]), np.max(centers[:, 1])], [trend_x[0] * np.min(centers[:, 1]) + trend_x[1], trend_x[0] * np.max(centers[:, 1]) + trend_x[1]], 
                       linestyle='dashed', color='r')
            plt.yticks(np.linspace(np.min(centers[:, 0]), np.max(centers[:, 0]), 5))
            plt.ylabel("x-axis")

            plt.sca(axes[1])
            plt.scatter(centers[:, 1], centers[:, 2])
            plt.plot([np.min(centers[:, 1]), np.max(centers[:, 1])], [trend_z[0] * np.min(centers[:, 1]) + trend_z[1], trend_z[0] * np.max(centers[:, 1]) + trend_z[1]], 
                       linestyle='dashed', color='r')
            plt.yticks(np.linspace(np.min(centers[:, 2]), np.max(centers[:, 2]), 5))
            plt.xticks(np.linspace(np.min(centers[:, 1]), np.max(centers[:, 1]), 7))
            plt.xlabel("y-axis")
            plt.ylabel("z-axis")

            plt.show()

        print(f'Axis fit rmse: {rmse}')
        self.axis_loc = c_axis
        self.spindle_axis = axis_dir
        self.spindle_cloud = spindle_bounded.T
        self.line_fit_rmse = rmse  # Store fit quality
        if show:
            pcd = Numpy_to_Open3D(self.spindle_cloud)
            #visualize_axis(pcd, c_axis, axis_dir, length=100)

#region SPINDLE FIT
    class Panel:
        def __init__(self, center, size, points):
            self.center = center  # tuple (x, y) coordinates
            self.size = size      # float, size of the panel
            self.points = points  # numpy array, shape (3, n_points)
            self.num_points = points.shape[1]  # number of points
            self.fit_params = None  # dict or None, fit parameters
            self.fit_points_2d = None
            self.good_fit_flag = True
            self.region_stddev = None
            self.weight = None

    def setup_coordinate_system(self):
        """Set up an orthogonal coordinate system based on the bar axis."""
        approx_axis = self.bar_axis
        if abs(approx_axis[0]) < min(abs(approx_axis[1]), abs(approx_axis[2])):
            u = np.array([1, 0, 0])
        elif abs(approx_axis[1]) < abs(approx_axis[2]):
            u = np.array([0, 1, 0])
        else:
            u = np.array([0, 0, 1])
        u = u - np.dot(u, approx_axis) * approx_axis
        u = u / np.linalg.norm(u)
        v = np.cross(approx_axis, u)
        return u, v

    def select_spindle_points(self, axial_cutoff, show_flag=False, spindle_cuts=None):
        """Select points along the bar axis below the axial cutoff."""
        projection = np.dot(self.cloud.T, self.bar_axis)
        if self.side == 'left':
            mask = (projection <= axial_cutoff)
        elif self.side == 'right':
            mask = (projection >= axial_cutoff)

        if show_flag:
            print('Showing spindle cloud after axial filter'); self.show_cloud(self.cloud.T[mask, :].T)

        # Compute z maximum and filter points according to it
        
        y_values = self.cloud.T[mask, 1]
        y_max = np.max(y_values)
        z_values = self.cloud.T[mask, 2]  
        z_max = np.max(z_values)

        if self.scan_type == 'sim':
            z_threshold_top = z_max + 1000
            z_threshold_bottom = -1000
            y_threshold_front = 1000
        elif self.scan_type == 'live':
            z_threshold_top = z_max - 25 #20
            z_threshold_bottom = z_threshold_top - 70 #45
            if self.side == 'right':
                y_threshold_front = y_max - 25
            elif self.side == 'left':
                y_threshold_front = y_max + 25

        z_mask = (z_values <= z_threshold_top) & (z_values >= z_threshold_bottom) & (y_values <= y_threshold_front)
        final_mask = mask.copy()
        final_mask[final_mask] = z_mask
        return self.cloud.T[final_mask, :]

    def project_to_xy(self, points_3d):
        """Project 3D points onto the global XY plane."""
        return points_3d[:, :2]  # Assumes x, y are first two coordinates

    def create_initial_grid(self, u_values, v_values, points_3d, box_size, overlap_factor):
        """
        Create a grid of panels and collect 3D points within each.

        Args:
            u_values (array): Axial coordinates in 2D plane.
            v_values (array): Tangential coordinates in 2D plane.
            points_3d (array): 3D points to be gridded.
            box_size (float): Size of square panels in mm.
            overlap_factor (float): Factor for panel overlap.

        Returns:
            tuple: (panels, (u_grid, v_grid), valid_panel_centers, bounds)
        """
        u_lower, u_upper = np.min(u_values), np.max(u_values)
        v_lower, v_upper = np.min(v_values), np.max(v_values)
        grid_spacing = box_size / overlap_factor
        self.grid_spacing = grid_spacing  # Store for later use
        half_space = grid_spacing / 2
        u_grid = np.arange(u_lower + half_space, u_upper, grid_spacing)
        v_grid = np.arange(v_lower + half_space, v_upper, grid_spacing)
        
        self.panels = []
        for u_i in u_grid:
            for v_j in v_grid:
                u_min = u_i - box_size / 2
                u_max = u_i + box_size / 2
                v_min = v_j - box_size / 2
                v_max = v_j + box_size / 2
                mask = (u_values >= u_min) & (u_values <= u_max) & \
                       (v_values >= v_min) & (v_values <= v_max)
                panel_points = points_3d[mask]
                panel_center = u_i * self.axis_xy + v_j * self.tangential_xy
                panel_obj = self.Panel(center=panel_center, size=box_size, points=panel_points.T)
                self.panels.append(panel_obj)
        bounds = (u_lower, u_upper, v_lower, v_upper)
        return (u_grid, v_grid), bounds

    def create_grid(self, current_size, points_3d):
        """
        Create a grid of panels with specified size.

        Args:
            current_size (float): Size of panels in mm.
            u_values (array): Axial coordinates in 2D plane.
            v_values (array): Tangential coordinates in 2D plane.
            points_3d (array): 3D points to be gridded.

        Returns:
            tuple: (list of panels, bounds as (u_lower, u_upper, v_lower, v_upper))
        """
        grid_spacing = current_size / self.overlap_factor
        u_grid = np.arange(self.u_lower + grid_spacing/2, self.u_upper, grid_spacing)
        v_grid = np.arange(self.v_lower + grid_spacing/2, self.v_upper, grid_spacing)
        panels = []
        for u_i in u_grid:
            for v_j in v_grid:
                u_min = u_i - current_size / 2
                u_max = u_i + current_size / 2
                v_min = v_j - current_size / 2
                v_max = v_j + current_size / 2
                mask = (self.u_values >= u_min) & (self.u_values <= u_max) & \
                    (self.v_values >= v_min) & (self.v_values <= v_max)
                panel_points = points_3d[mask]
                if panel_points.shape[0] > 50:
                    panel_center = u_i * self.axis_xy + v_j * self.tangential_xy
                    panel = self.Panel(center=panel_center, size=current_size, points=panel_points.T)
                    panels.append(panel)
        return panels

    def create_slices(self, current_size, points_3d):
        """
        Create a grid of panels with specified size.

        Args:
            current_size (float): Size of panels in mm.
            u_values (array): Axial coordinates in 2D plane.
            v_values (array): Tangential coordinates in 2D plane.
            points_3d (array): 3D points to be gridded.

        Returns:
            tuple: (list of panels, bounds as (u_lower, u_upper, v_lower, v_upper))
        """
        slice_spacing = current_size / self.overlap_factor
        u_grid = np.arange(self.u_lower + slice_spacing/2, self.u_upper, slice_spacing)
        panels = []
        for u_i in u_grid:
            u_min = u_i - current_size / 2
            u_max = u_i + current_size / 2
            mask = (self.u_values >= u_min) & (self.u_values <= u_max)
            panel_points = points_3d[mask]
            if panel_points.shape[0] > 50:
                v_centroid = np.mean(self.v_values[mask])
                panel_center = u_i * self.axis_xy + v_centroid * self.tangential_xy
                panel = self.Panel(center=panel_center, size=current_size, points=panel_points.T)
                panels.append(panel)
        return panels

    def create_region_grid(self, region_points, box_size, points_3d):
        grid_spacing = box_size / self.overlap_factor
        region_u_vals = self.project_to_xy(region_points) @ self.axis_xy
        region_v_vals = self.project_to_xy(region_points) @ self.tangential_xy
        u_lower, u_upper = np.min(region_u_vals), np.max(region_u_vals)
        v_lower, v_upper = np.min(region_v_vals), np.max(region_v_vals)
        u_grid = np.arange(u_lower + grid_spacing/2, u_upper, grid_spacing)
        v_grid = np.arange(v_lower + grid_spacing/2, v_upper, grid_spacing)
        panels = []
        for u_i in u_grid:
            for v_j in v_grid:
                u_min = u_i - box_size / 2
                u_max = u_i + box_size / 2
                v_min = v_j - box_size / 2
                v_max = v_j + box_size / 2
                mask = (self.u_values >= u_min) & (self.u_values <= u_max) & \
                       (self.v_values >= v_min) & (self.v_values <= v_max)
                panel_points = points_3d[mask]
                if panel_points.shape[0] > 50:
                    panel_center = u_i * self.axis_xy + v_j * self.tangential_xy
                    panel = self.Panel(center=panel_center, size=box_size, points=panel_points.T)
                    panels.append(panel)
        return panels

    def is_inside_good_panel(self, panel, good_panels):
        """
        Check if a panel's center lies within any existing good panel.

        Args:
            panel: Panel object to check.
            good_panels: List of panels with good fits.

        Returns:
            bool: True if panel center is inside any good panel, False otherwise.
        """
        for good_panel in good_panels:
            if not good_panel.good_fit_flag:
                continue
            center = panel.center
            good_center = good_panel.center
            half_size = good_panel.size / 2
            if (abs(center[0] - good_center[0]) < half_size and
                abs(center[1] - good_center[1]) < half_size):
                return True
        return False

    def create_sub_panels(self, u_values, v_values, points_3d, overlap_factor):
        new_panels = []
        for panel in self.panels:
            if panel.good_fit_flag:
                center_xy = panel.center
                center_u, center_v = center_xy @ self.axis_xy, center_xy @ self.tangential_xy
                center_uv = np.array([center_u, center_v])
                size = panel.size
                sub_size = size / 2
                half_size = size / 2
                half_sub_size = sub_size /2
                offsets = [
                    # Top row
                    (-half_size - half_sub_size, half_size + half_sub_size),
                    (-half_sub_size, half_size + half_sub_size),
                    (half_sub_size, half_size + half_sub_size),
                    (half_size + half_sub_size, half_size + half_sub_size),
                    # Bottom row
                    (-half_size - half_sub_size, -half_size - half_sub_size),
                    (-half_sub_size, -half_size - half_sub_size),
                    (half_sub_size, -half_size - half_sub_size),
                    (half_size + half_sub_size, -half_size - half_sub_size),
                    # Left column
                    (-half_size - half_sub_size, half_sub_size),
                    (-half_size - half_sub_size, -half_sub_size),
                    # Right column
                    (half_size + half_sub_size, half_sub_size),
                    (half_size + half_sub_size, -half_sub_size),]
                
                # print('center_xy', center_xy, 'center_uv', center_uv)
                for du, dv in offsets:
                    sub_center = center_uv + np.array([du, dv])
                    overlapped_size = half_sub_size*overlap_factor
                    u_min = sub_center[0] - overlapped_size
                    u_max = sub_center[0] + overlapped_size
                    v_min = sub_center[1] - overlapped_size
                    v_max = sub_center[1] + overlapped_size
                    # print(u_min, u_max, v_min, v_max)
                    mask = (u_values >= u_min) & (u_values <= u_max) & \
                           (v_values >= v_min) & (v_values <= v_max)
                    sub_points = points_3d[mask]
                    sub_center = sub_center[0] * self.axis_xy + sub_center[1] * self.tangential_xy
                    # print(mask.any() == True)
                    # print(sub_points)
                    if sub_points.T.shape[1] > 100:
                        sub_panel = self.Panel(center=sub_center, size=sub_size, points=sub_points.T)
                        new_panels.append(sub_panel)
        self.panels.extend(new_panels)

    def clean_overlapped_panels(self, box_size):
        for _ in range(5):
            for i, panel in enumerate(self.panels):
                for j, other_panel in enumerate(self.panels):
                    if i != j and other_panel.good_fit_flag:
                        center = panel.center
                        other_center = other_panel.center
                        half_size = other_panel.size / 2
                        if (abs(center[0] - other_center[0]) < half_size and
                            abs(center[1] - other_center[1]) < half_size):
                            if panel.size < box_size:
                                del self.panels[i]
                            break
        # for i in sorted(to_remove, reverse=True):
        #     del self.panels[i]

    def fit_cylinder_to_panel(self, panel, knn=10):
        """Fit a cylinder to a panel's points and optionally visualize the fit."""
        panel_points = panel.points
        pcd = Numpy_to_Open3D(panel_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        approximate_center = np.mean(panel.points.T, axis=0)
        normals = np.asarray(pcd.normals)
        for i, (point, normal) in enumerate(zip(np.asarray(pcd.points), normals)):
            to_center = approximate_center - point
            if np.dot(normal, to_center) > 0:  # Flip if pointing toward center
                normals[i] = -normal
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        normals = np.asarray(pcd.normals)
        M = np.sum([np.outer(n, n) for n in normals], axis=0)
        eigenvalues, eigenvectors = eigh(M)
        axis = eigenvectors[:, np.argmin(eigenvalues)]
        colinearity = np.abs(np.dot(axis, self.bar_axis))
        
        if abs(axis[0]) < min(abs(axis[1]), abs(axis[2])):
            u = np.array([1, 0, 0])
        elif abs(axis[1]) < abs(axis[2]):
            u = np.array([0, 1, 0])
        else:
            u = np.array([0, 0, 1])
        u = u - np.dot(u, axis) * axis
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        points_2d = np.dot(panel_points.T, np.array([u, v]).T)
        
        try:
            A = np.hstack([points_2d, np.ones((len(points_2d), 1))])
            b = -(points_2d[:, 0]**2 + points_2d[:, 1]**2)
            abc = np.linalg.lstsq(A, b, rcond=None)[0]
            a, b, c = abc
            discriminant = a**2 + b**2 - 4*c
            if discriminant <= 0:
                return None
            center_2d = [-a / 2, -b / 2]
            radius = np.sqrt((a/2)**2 + (b/2)**2 - c)
            radii = np.sqrt((points_2d[:, 0] - center_2d[0])**2 + (points_2d[:, 1] - center_2d[1])**2)
            stddev = np.std(radii)
            C = np.mean(panel_points.T, axis=0)
            t = np.dot(C, axis)
            center_3d = center_2d[0] * u + center_2d[1] * v + t * axis
                
            panel.fit_points_2d = points_2d
            panel.fit_params = {'axis': axis, 'colinearity': colinearity, 'center_3d': center_3d, 
                                'radius': radius, 'stddev': stddev, 'center_2d': center_2d}
        except np.linalg.LinAlgError:
            pass

    def visualize_panel_normals(self, panel, knn=10):
        """
        Visualizes a panel's points and their estimated normals.

        Args:
            panel: Panel object with points and normals attributes.
        """
        # Create point cloud from panel points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(panel.points.T)  # Shape (n_points, 3)
        
        # Estimate normals if not already computed
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
            approximate_center = np.mean(panel.points.T, axis=0)
            normals = np.asarray(pcd.normals)
            for i, (point, normal) in enumerate(zip(np.asarray(pcd.points), normals)):
                to_center = approximate_center - point
                if np.dot(normal, to_center) > 0:  # Flip if pointing toward center
                    normals[i] = -normal
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Panel Normals")
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Configure rendering options to show normals
        opt = vis.get_render_option()
        opt.point_show_normal = True  # Display normals
        opt.point_size = 3.0  # Point size for visibility
        
        # Run visualization
        vis.run()
        vis.destroy_window()

    def fit_axis_to_weighted_spindle_panels(self):
        """
        Fits the spindle axis using weighted panels, where each panel's weight applies to all its points.
        
        Returns:
            axis: The computed spindle axis as a unit vector.
        """
        # Collect points and weights from panels with non-None weights
        all_points = []
        all_weights = []
        for region_idx, panels in enumerate(self.raw_regions_panels):
            for panel in panels:
                if not panel.weight == None:  # Check for non-None weight
                    points = panel.points.T  # Transpose assumes points are in columns
                    weight = panel.weight
                    all_points.append(points)
                    all_weights.append(np.full(points.shape[0], weight))
        
        if not all_points:
            raise ValueError("No panels with non-None weights found.")
        
        # Concatenate all points and weights
        all_points = np.vstack(all_points)
        all_weights = np.concatenate(all_weights)
        
        # Create Open3D point cloud and estimate normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
        normals = np.asarray(pcd.normals)
        
        # Compute weighted M matrix
        M = np.zeros((3, 3))
        for i in range(len(normals)):
            n = normals[i]
            w = all_weights[i]
            M += w * np.outer(n, n)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(M)
        axis = eigenvectors[:, np.argmin(eigenvalues)]
        
        # Ensure axis is a unit vector
        axis = axis / np.linalg.norm(axis)
        return axis

    def fit_axis_to_weighted_spindle_panels2(self, knn=10, view_normals=False):
        """
        Fits the spindle axis using weighted panels, where each panel's weight applies to all its points.
        
        Returns:
            axis: The computed spindle axis as a unit vector.
        """
        # Collect points and weights from panels with non-None weights
        all_points = []
        all_weights = []
        centers = []
        for panel in self.panel_groups:
            if not panel.weight == None:  # Check for non-None weight
                points = panel.points.T  # Transpose assumes points are in columns
                weight = panel.weight; #print('weight:', weight)
                all_points.append(points)
                all_weights.append(np.full(points.shape[0], weight))
            centers.append(panel.fit_params['center_3d'])
        
        if not all_points:
            raise ValueError("No panels with non-None weights found.")
        
        # Concatenate all points and weights
        all_points = np.vstack(all_points)
        all_weights = np.concatenate(all_weights)
        
        # Create Open3D point cloud and estimate normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        approximate_center = np.mean(centers, axis=0)
        normals = np.asarray(pcd.normals)
        for i, (point, normal) in enumerate(zip(np.asarray(pcd.points), normals)):
            to_center = approximate_center - point
            if np.dot(normal, to_center) > 0:  # Flip if pointing toward center
                normals[i] = -normal
        pcd.normals = o3d.utility.Vector3dVector(normals)
        normals = np.asarray(pcd.normals)

        if view_normals:
            # Create visualization window
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Panel Normals")
            
            # Add point cloud
            vis.add_geometry(pcd)
            
            # Configure rendering options to show normals
            opt = vis.get_render_option()
            opt.point_show_normal = True  # Display normals
            opt.point_size = 3.0  # Point size for visibility
            
            # Run visualization
            vis.run()
            vis.destroy_window()
        
        # Compute weighted M matrix
        M = np.zeros((3, 3))
        for i in range(len(normals)):
            n = normals[i]
            w = all_weights[i]
            M += w * np.outer(n, n)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(M)
        axis = eigenvectors[:, np.argmin(eigenvalues)]
        
        # Ensure axis is a unit vector
        axis = axis / np.linalg.norm(axis)
        return axis

    def visualize_grid(self, panels, panel_size):
        """Visualize the 2D projection with grid overlay."""
        plt.scatter(self.points_xy[:, 0], self.points_xy[:, 1], s=1, label='Points')
        for panel in panels:
            if int(panel.size) == int(panel_size):
                center = panel.center
                half_size = panel.size / 2
                edgecolor = 'orange'
                rect = plt.Rectangle((center[0] - half_size, center[1] - half_size),
                                    panel.size, panel.size,
                                    fill=False, edgecolor=edgecolor, linewidth=0.5)
                plt.gca().add_patch(rect)
        u_lower_xy = self.u_lower * self.axis_xy
        u_upper_xy = self.u_upper * self.axis_xy
        v_lower_xy = self.v_lower * self.tangential_xy
        v_upper_xy = self.v_upper * self.tangential_xy
        corners = [
            u_lower_xy + v_lower_xy,
            u_lower_xy + v_upper_xy,
            u_upper_xy + v_upper_xy,
            u_upper_xy + v_lower_xy
        ]
        corners = np.array(corners)
        plt.plot(
            np.append(corners[:, 0], corners[0, 0]),
            np.append(corners[:, 1], corners[0, 1]),
            c='black', linewidth=0.5
        )
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.axis('equal')
        plt.title("2D Projection of Spindle Points with Grid (Global XY)")
        plt.legend()
        plt.show()

    def visualize_good_panels(self, panels=None):
        """Visualize the 2D projection highlighting panels with best cylinder fits."""
        plt.figure()
        plt.scatter(self.points_xy[:, 0], self.points_xy[:, 1], s=1, label='Points')
        
        # Plot rectangles for best-fit panels only
        if panels == None:
            use_panels = self.panels
        else:
            use_panels = panels

        if  hasattr(use_panels, '__iter__'):
            for panel in use_panels:
                if panel.good_fit_flag == True:
                    center = panel.center
                    half_size = panel.size / 2
                    edgecolor = 'orange'
                    plt.scatter(center[0], center[1], c='red', s=20)
                    rect = plt.Rectangle(
                        (center[0] - half_size, center[1] - half_size),
                        panel.size, panel.size,
                        fill=False, edgecolor=edgecolor, linewidth=0.5
                    )
                    plt.gca().add_patch(rect)
        else:
            center = use_panels.center
            half_size = use_panels.size / 2
            edgecolor = 'green'
            plt.scatter(center[0], center[1], c='red', s=20)
            rect = plt.Rectangle(
                (center[0] - half_size, center[1] - half_size),
                use_panels.size, use_panels.size,
                fill=False, edgecolor=edgecolor, linewidth=0.5
            )
            plt.gca().add_patch(rect)
        
        # Draw bounding box
        u_lower_xy = self.u_lower * self.axis_xy
        u_upper_xy = self.u_upper * self.axis_xy
        v_lower_xy = self.v_lower * self.tangential_xy
        v_upper_xy = self.v_upper * self.tangential_xy
        corners = [
            u_lower_xy + v_lower_xy,
            u_lower_xy + v_upper_xy,
            u_upper_xy + v_upper_xy,
            u_upper_xy + v_lower_xy
        ]
        corners = np.array(corners)
        plt.plot(
            np.append(corners[:, 0], corners[0, 0]),
            np.append(corners[:, 1], corners[0, 1]),
            c='black', linewidth=0.5
        )
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.axis('equal')
        plt.title("2D Projection of Spindle Points with Best-Fit Panels (Global XY)")
        plt.legend()
        plt.show()

    def plot_panel_fit(self, panel, show_normals=False):
        """
        Plots a panel's points and its fitted cylinder surface in 3D using Matplotlib.

        Parameters:
            panel_points (array): 3D points of the panel, shape (3, n_points).
            cylinder_params (dict): Cylinder parameters {'axis', 'center_3d', 'radius', 'stddev'} or None.
        """
        if panel.fit_params is None:
            return
        axis = panel.fit_params['axis']
        center_3d = panel.fit_params['center_3d']
        radius = panel.fit_params['radius']
        stddev = panel.fit_params['stddev']
        center_2d = panel.fit_params['center_2d']
        colinearity = panel.fit_params['colinearity']
        points_2d = panel.fit_points_2d
        size = panel.size
        
        #region PLOT 2D FIT
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = center_2d[0] + radius * np.cos(theta)
        y_circle = center_2d[1] + radius * np.sin(theta)
        fig1 = plt.figure(figsize=(6, 6))
        fig1.canvas.manager.window.wm_geometry("600x600+100+100")  # 600x600 size, position (100, 100)
        plt.plot(x_circle, y_circle, 'r-', label='Best-fit circle', linewidth=0.5)
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1)
        plt.scatter(center_2d[0], center_2d[1])
        plt.title(f"stddev: {stddev:.4f}, radius: {radius:.2f}, center_xy: {center_2d[0]:.2f}, {center_2d[1]:.2f}")
        plt.axis('equal')
        plt.xlabel("u-axis")
        plt.ylabel("v-axis")
        #endregion
        
        #region PLOT 3D FIT
        fig = plt.figure(figsize=(6, 6))
        fig.canvas.manager.window.wm_geometry("600x600+700+100")  # 600x600 size, position (700, 100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(panel.points[0, :], panel.points[1, :], panel.points[2, :], s=1, c='blue', label='Points')
        
        if abs(axis[0]) < min(abs(axis[1]), abs(axis[2])):
            u = np.array([1, 0, 0])
        elif abs(axis[1]) < abs(axis[2]):
            u = np.array([0, 1, 0])
        else:
            u = np.array([0, 0, 1])
        u = u - np.dot(u, axis) * axis
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        t_values = np.dot(panel.points.T - center_3d, axis)
        t_range = np.max(t_values) - np.min(t_values)
        t_half = 0.5 * t_range
        t = np.linspace(-t_half, t_half, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta, t = np.meshgrid(theta, t)
        X = center_3d[0] + radius * np.cos(theta) * u[0] + radius * np.sin(theta) * v[0] + t * axis[0]
        Y = center_3d[1] + radius * np.cos(theta) * u[1] + radius * np.sin(theta) * v[1] + t * axis[1]
        Z = center_3d[2] + radius * np.cos(theta) * u[2] + radius * np.sin(theta) * v[2] + t * axis[2]
        ax.plot_surface(X, Y, Z, color='red', alpha=0.3, rstride=5, cstride=5, label='Cylinder')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Panel Points and Cylinder Fit\nSize: {size:.1f}mm, Bar Colinearity: {colinearity:.4f}')
        #endregion
        
        plt.show()

    def group_panels_by_radius(self, radius_tolerance=0.1, max_radius=50):
        """
        Combines points from panels with similar radii into a single panel per radius group.

        Args:
            radius_tolerance (float): Maximum radius difference (mm) to consider two radii the same.

        Stores:
            self.panel_groups: List of combined panels with radius attribute set.
        """
        self.panel_groups = []
        
        # Filter panels with valid radius in fit_params
        valid_panels = [panel for panel in self.panels if panel.fit_params and 'radius' in panel.fit_params]
        if not valid_panels:
            return
        
        # Sort panels by radius
        sorted_panels = sorted(valid_panels, key=lambda p: p.fit_params['radius'])
        
        # Handle single panel case
        if len(sorted_panels) == 1:
            sorted_panels[0].radius = sorted_panels[0].fit_params['radius']
            self.panel_groups.append(sorted_panels[0])
            return
        
        # Extract radii and compute step sizes
        radii = [p.fit_params['radius'] for p in sorted_panels]
        
        # Group panels based on step sizes
        groups = []
        current_group = [sorted_panels[0]]
        previous_radius = radii[0]
        
        for panel in sorted_panels[1:]:
            current_radius = panel.fit_params['radius']
            step = current_radius - previous_radius
            
            # If step exceeds tolerance, start a new group
            if step > radius_tolerance:
                groups.append(current_group)
                current_group = [panel]
            else:
                current_group.append(panel)
            
            previous_radius = current_radius
        
        # Append the last group
        if current_group:
            groups.append(current_group)
        
        # Create single panel per group by combining points
        for group in groups:
            group_radii = [p.fit_params['radius'] for p in group]
            avg_radius = np.mean(group_radii)
            
            # Combine points from all panels in the group
            combined_points = np.concatenate([p.points for p in group], axis=1)
            
            # Calculate center as average of panel centers
            combined_center = np.mean([p.center for p in group], axis=0)
            
            # Use maximum size from panels in the group
            combined_size = max(p.size for p in group)
            
            # Create new panel with combined points
            combined_panel = self.Panel(center=combined_center, size=combined_size, points=combined_points)
            combined_panel.weight = 1.0
            self.fit_cylinder_to_panel(combined_panel)
            
            if combined_points.shape[1] > 40000:
                half_points = combined_points.shape[1] // 2
                first_half = self.Panel(center=combined_center, size=combined_size, points=combined_points[:, :half_points])
                second_half = self.Panel(center=combined_center, size=combined_size, points=combined_points[:, half_points:])
                self.fit_cylinder_to_panel(first_half)
                self.fit_cylinder_to_panel(second_half)
                if first_half.fit_params['radius'] <= max_radius:
                    self.panel_groups.append(first_half)
                if second_half.fit_params['radius'] <= max_radius:
                    self.panel_groups.append(second_half)
            else:
                if combined_panel.fit_params['radius'] <= max_radius:
                    self.panel_groups.append(combined_panel)

    def group_panels_by_axial_location(self, y_thresholds=None, min_gap=20.0):
        """
        Groups panels by their axial location along the bar axis (self.bar_axis).
        Panels are sorted by their projection onto self.bar_axis and grouped based on
        either predefined thresholds or detected gaps in the axial coordinates.

        Args:
            y_thresholds (list, optional): List of axial coordinates to use as thresholds for grouping.
                                        If None, groups are determined by detecting gaps.
            min_gap (float, optional): Minimum gap size (in mm) in axial coordinates to consider
                                    as a separation between groups. Used if y_thresholds is None.

        Stores:
            self.axial_groups: Dict with group indices as keys and lists of panels as values.
        """
        # Filter panels with valid centers and good fits
        valid_panels = [panel for panel in self.panels if panel.good_fit_flag and panel.center is not None]
        if not valid_panels:
            self.axial_groups = {}
            return

        # Project panel centers onto bar axis to get axial coordinates
        axial_coords = [np.dot(panel.center, self.axis_xy) for panel in valid_panels]
        panel_indices = list(range(len(valid_panels)))

        # Sort panels by axial coordinate (ascending)
        sorted_indices = np.argsort(axial_coords)
        sorted_axial_coords = [axial_coords[i] for i in sorted_indices]
        sorted_panels = [valid_panels[i] for i in sorted_indices]

        gaps = np.diff(sorted_axial_coords)
        avg_gap = np.mean(gaps)

        self.axial_groups = {}
        group_idx = 0

        if y_thresholds is not None:
            # Use predefined thresholds (projected onto bar axis)
            y_thresholds = sorted([np.dot(thresh, self.bar_axis) for thresh in y_thresholds])
            current_group = []
            threshold_idx = 0

            for i, (axial, panel) in enumerate(zip(sorted_axial_coords, sorted_panels)):
                while threshold_idx < len(y_thresholds) and axial >= y_thresholds[threshold_idx]:
                    if current_group:
                        self.axial_groups[group_idx] = current_group
                        group_idx += 1
                    current_group = []
                    threshold_idx += 1
                current_group.append(panel)

            if current_group:
                self.axial_groups[group_idx] = current_group

        else:
            # Automatically detect groups by finding gaps in axial coordinates
            current_group = [sorted_panels[0]]
            for i in range(1, len(sorted_panels)):
                gap = sorted_axial_coords[i] - sorted_axial_coords[i-1]
                if gap > avg_gap*2.5:
                    self.axial_groups[group_idx] = current_group
                    group_idx += 1
                    current_group = [sorted_panels[i]]
                else:
                    current_group.append(sorted_panels[i])

            if current_group:
                self.axial_groups[group_idx] = current_group

        # Log the number of groups and panels in each
        # for idx, group in self.axial_groups.items():
        #     print(f"Group {idx}: {len(group)} panels, Axial-range: "
        #         f"{min(np.dot(p.center, self.axis_xy) for p in group):.2f} to "
        #         f"{max(np.dot(p.center, self.axis_xy) for p in group):.2f} mm")
            
        # for idx, group in self.axial_groups.items():
        #     radii = [p.fit_params['radius'] for p in group if p.fit_params and 'radius' in p.fit_params]
        #     avg_radius = np.mean(radii) if radii else 0.0
        #     print(f"Group {idx}: Avg Radius = {avg_radius:.2f} mm")

    def combine_panels_into_single_panel(self, group):
        """
        Combines all panels in a group into a single panel with combined 3D points
        and centroid as the center. No fit parameters are calculated.

        Args:
            group (list): List of panel objects to combine.

        Returns:
            Panel: A new panel object with combined points and centroid center.
        """
        if not group:
            return None

        # Combine all 3D points from the group
        combined_points = np.hstack([panel.points for panel in group])

        # Calculate x-y centroid as the center
        x_coords = combined_points[0]
        y_coords = combined_points[1]
        centroid_xy = np.array([np.mean(x_coords), np.mean(y_coords), 0.0])

        # Create new panel with combined points and centroid center
        new_panel = self.Panel(center=centroid_xy, size=0.0, points=combined_points)
        new_panel.num_points = combined_points.shape[1]

        return new_panel

    def combine_similar_regions(self):
        """
        Combines adjacent regions in self.regions whose radii are within 1mm of each other.
        Updates self.regions with combined panels.
        """
        i = 0
        while i < len(self.regions) - 1:
            radius_i = self.regions[i].fit_params['radius'] if self.regions[i].fit_params and 'radius' in self.regions[i].fit_params else float('inf')
            radius_j = self.regions[i + 1].fit_params['radius'] if self.regions[i + 1].fit_params and 'radius' in self.regions[i + 1].fit_params else float('inf')
            if abs(radius_i - radius_j) <= 1.0:
                combined_points = np.hstack((self.regions[i].points, self.regions[i + 1].points))
                x_coords = combined_points[0]
                y_coords = combined_points[1]
                centroid_xy = np.array([np.mean(x_coords), np.mean(y_coords), 0.0])
                new_region = self.Panel(center=centroid_xy, size=0.0, points=combined_points)
                new_region.num_points = combined_points.shape[1]
                self.regions[i] = new_region
                self.regions.pop(i + 1)
            else:
                i += 1

    def get_all_raw_points_in_region(self, region, points_3d):
        plt.scatter(region.points[0], region.points[1], s=2)
        
        u_vals, v_vals = self.u_values, self.v_values
        region_u_vals = self.project_to_xy(region.points.T) @ self.axis_xy
        region_v_vals = self.project_to_xy(region.points.T) @ self.tangential_xy

        u_lower, u_upper = np.min(region_u_vals), np.max(region_u_vals)
        v_lower, v_upper = np.min(region_v_vals), np.max(region_v_vals)
        mask = (u_vals >= u_lower) & (u_vals <= u_upper) & \
               (v_vals >= v_lower) & (v_vals <= v_upper)
        raw_points = points_3d[mask]
        plt.scatter(raw_points.T[0], raw_points.T[1], s=1)
        # plt.show()
    
        return raw_points
        
    def weight_panels(self):
        """
        Assigns weights to panels in each region based on alignment with the region's cylinder
        and the distribution of radial stddev values.

        For each panel:
        - Fits a cylinder and checks alignment with the region's cylinder axis.
        - If misaligned (>5 degrees), assigns weight 0.
        - If aligned, computes stddev of radial errors relative to the region's cylinder.
        - Assigns weights based on stddev percentiles:
        - stddev ≤ 11th percentile: weight = 1
        - stddev ≥ 89th percentile: weight = 0
        - 11th < stddev < 89th: weight = (1 - 2x)^3 + 0.5, where x is normalized percentile rank
        """
        for region_idx, region in enumerate(self.regions):
            # Skip if region has no valid cylinder fit
            if region.fit_params is None or 'axis' not in region.fit_params:
                continue
            
            # Get region's cylinder parameters
            A_region = region.fit_params['axis']
            C_region = region.fit_params['center_3d']
            R_region = region.fit_params['radius']
            panels = self.raw_regions_panels[region_idx]
            
            # Ensure cylinders are fitted to panels
            for panel in panels:
                if not hasattr(panel, 'fit_params') or panel.fit_params is None:
                    self.fit_cylinder_to_panel(panel)
            
            # Compute stddev for each panel and collect values
            stddev_list = []
            for panel in panels:
                if panel.fit_params is None or 'axis' not in panel.fit_params:
                    panel.weight = 0.0
                    continue
                
                # Check axis alignment
                if panel.fit_params['colinearity'] < 0.98:  # Threshold for "closely aligned"
                    panel.weight = 0.0
                    continue
                
                # Compute radial stddev using region's cylinder
                points = panel.points.T  # Shape (n_points, 3)
                delta = points - C_region
                proj = np.dot(delta, A_region)
                Q = C_region + proj[:, np.newaxis] * A_region
                distances = np.linalg.norm(points - Q, axis=1)
                stddev = np.std(distances)
                panel.region_stddev = stddev
                stddev_list.append(stddev)
            
            # Assign weights based on stddev distribution
            if stddev_list:
                P00 = np.percentile(stddev_list, 0)
                P100 = np.percentile(stddev_list, 100)
                for panel in panels:
                    if not panel.region_stddev == None:
                        # Normalize x between 0 and 1
                        x = (panel.region_stddev) / (P100 - P00)
                        panel.weight = max(0.0, min(1.0, -1.25*x + 1.125)) # Accepcts all below 10th, rejects all above 90th. Linear between
                    # print(panel.region_stddev, panel.weight)
                    # self.plot_panel_fit(panel)
            else:
                # If no panels passed alignment, assign weight 0
                for panel in panels:
                    panel.weight = 0.0

    def filter_outliers_in_panel(self, panel, iqr_scales=[1.0, 0.5, 0.2], keep_inner=False):
        """
        Iteratively filters out outlier points in the panel based on the best-fit cylinder.

        Args:
            panel: The panel containing points to be filtered.
            iqr_scales (list): List of IQR scales for iterative filtering (e.g., [1.0, 0.5]).
        """
        points = panel.points.T  # Assuming points are in columns, transpose to rows
        
        for iqr_scale in iqr_scales:
            # Fit cylinder to current points
            if len(points) < 30:
                panel.good_fit_flag = False
                return
            xy_centroid = np.mean(points[:, :2], axis=0)
            x_size = np.max(points[:, 0]) - np.min(points[:, 0])
            temp_panel = self.Panel(center=xy_centroid, size=x_size, points=points.T)
            # print(f'Fitting cylinder to {temp_panel.num_points} points')
            self.fit_cylinder_to_panel(temp_panel)
            
            if keep_inner:
                if temp_panel.fit_params['stddev'] <= 0.5:
                    # Update the panel's points with inliers
                    panel.points = points.T
                    panel.num_points = panel.points.shape[1]
                    print(f'Fitting cylinder to final {panel.num_points} points')
                    self.fit_cylinder_to_panel(panel)
                    return
                # self.plot_panel_fit(temp_panel)
            if temp_panel.fit_params['stddev'] <= 0.10:
                # Update the panel's points with inliers
                panel.points = points.T
                panel.num_points = panel.points.shape[1]
                print(f'Fitting cylinder to final {panel.num_points} points')
                self.fit_cylinder_to_panel(panel)
                return
            
            # Extract cylinder parameters
            A = temp_panel.fit_params['axis']         # Cylinder axis
            C = temp_panel.fit_params['center_3d']    # Cylinder center
            R = temp_panel.fit_params['radius']       # Cylinder radius
            
            # Compute radial errors
            delta = points - C
            proj = np.dot(delta, A)
            Q = C + proj[:, np.newaxis] * A
            distance_to_axis = np.linalg.norm(points - Q, axis=1)
            radial_errors = np.abs(distance_to_axis - R)
            
            # Compute IQR and determine threshold
            Q1, Q3 = np.percentile(radial_errors, [25, 75])
            IQR = Q3 - Q1
            upper_bound = Q3 + iqr_scale * IQR
            
            # Keep points where radial_error <= upper_bound
            mask = radial_errors <= upper_bound
            if keep_inner:
                radial_errors = distance_to_axis - R
                mask = (radial_errors <= upper_bound*1.0) & (distance_to_axis <= 65)
            points = points[mask]

            
        
        # Update the panel's points with inliers
        panel.points = points.T
        panel.num_points = panel.points.shape[1]
        
        # Refit the cylinder to the final points for consistency
        print(f'Fitting cylinder to final {panel.num_points} points')
        self.fit_cylinder_to_panel(panel)

    def fit_spindle_3D(self, axial_cutoff=-150, show_flag=False, plot_flag=False, box_size=50.0, spindle_cuts=[0, 500, 0, 500], overlap_factor=1.1, max_radius=50):
        """
        Fits a 3D spindle by creating grids of decreasing panel sizes, keeping good fits.

        Args:
            axial_cutoff (float): Threshold along the bar axis to isolate spindle points.
            show_flag (bool): If True, displays intermediate plots.
            box_size (float): Initial panel size in mm.
            min_size (float): Minimum panel size to stop subdivision.
            overlap_factor (float): Factor to scale panel sizes for point inclusion.
        """
        #region ITERATIVE GRID SEARCH FOR LOW NOISE
        # Store overlap_factor for use in create_grid
        self.overlap_factor = overlap_factor
        col_tol = 0.99  # 0.99
        
        # Setup coordinate system and project points
        spindle_points = self.select_spindle_points(axial_cutoff, show_flag=show_flag, spindle_cuts=spindle_cuts)
        self.spindle_cloud = spindle_points
        # if show_flag:
        #     print('Showing spindle cloud after all filters'); self.show_cloud(spindle_points.T)
        self.points_xy = self.project_to_xy(spindle_points)
        
        self.axis_xy = self.bar_axis[:2]
        if np.linalg.norm(self.axis_xy) > 1e-6:
            self.axis_xy = self.axis_xy / np.linalg.norm(self.axis_xy)
        else:
            self.axis_xy = np.array([1.0, 0.0])
        self.tangential_xy = np.array([-self.axis_xy[1], self.axis_xy[0]])
        
        self.u_values = self.points_xy @ self.axis_xy
        self.v_values = self.points_xy @ self.tangential_xy
        self.u_lower, self.u_upper = np.min(self.u_values), np.max(self.u_values)
        self.v_lower, self.v_upper = np.min(self.v_values), np.max(self.v_values)
        
        # Initialize with first grid
        self.panels = self.create_slices(box_size, spindle_points)
        # print(f'Showing intial grid of size {box_size}'); self.visualize_grid(panels=self.panels, panel_size=box_size)
        
        # Fit cylinders and filter bad sections
        for panel in self.panels:
            self.fit_cylinder_to_panel(panel)
            if panel.fit_params['stddev'] > 2.0 and panel.fit_params['colinearity'] >= col_tol-0.04:
                # self.plot_panel_fit(panel)
                self.filter_outliers_in_panel(panel, iqr_scales=[1.0, 0.8, 0.8], keep_inner=True)
                # print(f'Showing filtered panel in {box_size}mm')
                # self.plot_panel_fit(panel)

        stddevs = [panel.fit_params['stddev'] for panel in self.panels if panel.fit_params]
        stddev_cutoff = np.percentile(stddevs, 80) if stddevs else float('inf')
        self.panels = [panel for panel in self.panels  
                    if panel.fit_params and 
                    panel.fit_params['stddev'] <= stddev_cutoff and 
                    panel.fit_params['colinearity'] >= col_tol and
                    panel.fit_params['radius'] <= max_radius]
        # print(f'Showing initial good panels down to {box_size}mm'); self.visualize_good_panels()
        # for panel in self.panels:
        #     self.plot_panel_fit(panel)

        good_panels = self.panels.copy()
        new_panels = self.create_slices(box_size/2, spindle_points)
        # print(f'Showing new grid of size {box_size/2}'); self.visualize_grid(panels=new_panels, panel_size=box_size/2)
        
        new_panels = [panel for panel in new_panels 
                        if not self.is_inside_good_panel(panel, good_panels)]
        # print(f'Showing filtered grid of size {box_size/2}'); self.visualize_grid(panels=new_panels, panel_size=box_size/2)
        for panel in new_panels:
            self.fit_cylinder_to_panel(panel)
            # if panel.fit_params['stddev'] > 2.0 and panel.fit_params['colinearity'] >= 0.95:
            #     self.plot_panel_fit(panel)
            #     self.filter_outliers_in_panel(panel, iqr_scales=[1.5, 1.5], keep_inner=True)
            #     print(f'Showing filtered panel in {box_size/2}mm')
            #     self.plot_panel_fit(panel)
        new_good_panels = [panel for panel in new_panels 
                            if panel.fit_params and  
                            panel.fit_params['stddev'] <= stddev_cutoff*1.5 and
                            panel.fit_params['colinearity'] >= col_tol and
                            panel.fit_params['radius'] <= max_radius]
        self.panels.extend(new_good_panels)

        good_panels = self.panels.copy()
        new_panels = self.create_slices(box_size/3, spindle_points)
        # print(f'Showing new grid of size {box_size/3}'); self.visualize_grid(panels=new_panels, panel_size=box_size/3)
        
        new_panels = [panel for panel in new_panels 
                        if not self.is_inside_good_panel(panel, good_panels)]
        # print(f'Showing filtered grid of size {box_size/3}'); self.visualize_grid(panels=new_panels, panel_size=box_size/3)
        for panel in new_panels:
            self.fit_cylinder_to_panel(panel)
        new_good_panels = [panel for panel in new_panels 
                            if panel.fit_params and  
                            panel.fit_params['stddev'] <= stddev_cutoff*2.0 and
                            panel.fit_params['colinearity'] >= col_tol and
                            panel.fit_params['radius'] <= max_radius]
        self.panels.extend(new_good_panels)


        # if plot_flag:
        #     print(f'Showing good panels down to {box_size/3}mm'); self.visualize_good_panels()
        # for panel in self.panels:
        #     self.plot_panel_fit(panel)
        #endregion

        #region GROUP SLICES BY RADIUS AND SELECT GOOD POINTS
        self.group_panels_by_radius(radius_tolerance=0.10, max_radius=50)
        # if plot_flag:
        #     print('Showing slices grouped by radius'); self.visualize_good_panels(self.panel_groups)
        # self.fit_axis_to_weighted_spindle_panels2(knn=80, view_normals=True)

        for panel in self.panel_groups:
            # print('Showing original panel'); self.plot_panel_fit(panel)
            self.filter_outliers_in_panel(panel)
            # print('Showing filtered panel'); self.plot_panel_fit(panel)
            # self.visualize_panel_normals(panel, knn=10)
            # self.visualize_panel_normals(panel, knn=30)

        for panel in self.panel_groups:
            # Radius weight: 0 at radius < 7, 1 at radius > 25, linear in between
            radius_weight = np.clip((panel.fit_params['radius'] - 7) / (25 - 7), 0, 1)
            # Stddev weight: 1 at stddev < 0.4, 0 at stddev > 0.8, linear in between
            stddev_weight = np.clip((0.8 - panel.fit_params['stddev']) / (0.8 - 0.4), 0, 1)
            panel.weight = radius_weight * stddev_weight

        self.spindle_axis = self.fit_axis_to_weighted_spindle_panels2(knn=120, view_normals=False); print('Fitting axis to all good panels')
        #endregion

        #region 2D SLICES FIT
        approx_axis = self.spindle_axis
        good_spindle_points = self.combine_panels_into_single_panel(self.panel_groups).points
        if show_flag:
            print('Showing filtered spindle points'); self.show_cloud(good_spindle_points)
        if abs(approx_axis[0]) < min(abs(approx_axis[1]), abs(approx_axis[2])):
            u = np.array([1, 0, 0])
        elif abs(approx_axis[1]) < abs(approx_axis[2]):
            u = np.array([0, 1, 0])
        else:
            u = np.array([0, 0, 1])
        u = u - np.dot(u, approx_axis) * approx_axis
        u = u / np.linalg.norm(u)
        v = np.cross(approx_axis, u)

        # Project spindle points onto bar axis and slice into bins
        num_bins = 150
        min_points_per_bin = 30
        circle_fit_tol = 1.0
        t = np.dot(good_spindle_points.T, approx_axis)
        t_min, t_max = np.min(t), np.max(t)
        bin_edges = np.linspace(t_min, t_max, num_bins + 1)
        centers = []
        count = 0
        for i in range(num_bins):
            mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
            if np.sum(mask) < min_points_per_bin:
                continue
            points_bin = good_spindle_points.T[mask]
            
            # Project points in bin onto the plane orthogonal to bar axis
            points_2d = np.dot(points_bin, np.array([u, v]).T)
            maxC = np.max(points_2d)
            A = np.hstack([points_2d, np.ones((len(points_2d), 1))])
            b = -(points_2d[:, 0]**2 + points_2d[:, 1]**2)
            
            # Fit a circle to projected points and record its center in global 3D frame
            try:
                abc = np.linalg.lstsq(A, b, rcond=None)[0]
                a, b, c = abc
                discriminant = a**2 + b**2 - 4*c
                if discriminant > 0:
                    center_2d = [-a / 2, -b / 2]
                    radius = np.sqrt((a/2)**2 + (b/2)**2 - c)
                    radii = np.sqrt((points_2d[:, 0] - center_2d[0])**2 + (points_2d[:, 1] - center_2d[1])**2)
                    stddev = np.std(radii)
                    if stddev < circle_fit_tol:
                        count += 1
                        t_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                        center_3d = t_center * approx_axis + center_2d[0] * u + center_2d[1] * v
                        centers.append(center_3d)
            except np.linalg.LinAlgError:
                continue
            if plot_flag:# and i < 10:
                # self.show_cloud(points_bin.T)
                theta = np.linspace(0, 2 * np.pi, 100)
                # x_circle = center_2d[0] + radius * np.cos(theta)
                # y_circle = center_2d[1] + radius * np.sin(theta)
                # plt.plot(x_circle, y_circle, 'r-', label='Best-fit circle', linewidth=0.5)
                # plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1)
                # plt.scatter(center_2d[0], center_2d[1])
                # plt.title(f"Projected Slice {i}. stddev: {stddev}")
                # plt.xlim(-maxC, maxC)
                # plt.ylim(-maxC, maxC)
                # plt.axis('equal')
                # plt.xlabel("u-axis")
                # plt.ylabel("v-axis")
                # plt.show()
        
        # Fit a line to 3D centers and compute standard deviation of distances to axis
        def fit_axis(centers, approx_axis):
            if len(centers) < 2:
                raise ValueError("Not enough valid circle fits to determine the axis.")
            centers = np.array(centers)
            # Compute centroid of centers as axis location
            c_axis = np.mean(centers, axis=0)
            # Perform PCA to find principal axis direction
            U, S, Vt = np.linalg.svd(centers - c_axis, full_matrices=False)
            axis_dir = Vt[0]  # Principal component (largest singular value)
            # Ensure axis positive direction aligns with approximate axis
            if np.dot(axis_dir, approx_axis) < 0:
                axis_dir = -axis_dir
            return c_axis, axis_dir

        # Filter outliers based on IQR multiplier
        def filter_outliers(centers, c_axis, axis_dir, iqr_multiplier):
            # Calculate perpendicular distances from centers to the fitted axis
            projections = np.dot(centers - c_axis, axis_dir)
            points_on_line = c_axis + np.outer(projections, axis_dir)
            distances = np.linalg.norm(centers - points_on_line, axis=1)
            stddev = np.std(distances)
            print(f'Centers stddev: {stddev:.3f}')
            if self.scan_type == 'sim':
                tol = 0.01
            elif self.scan_type == 'live':
                tol = 0.025
            if stddev <= tol or len(centers)/len(centers_orig) <= 0.75:
                return centers
            # Compute IQR and bounds for outlier filtering
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            threshold = iqr * iqr_multiplier
            median = np.median(distances)
            # Keep centers within median ± threshold
            mask = (np.abs(distances - median) <= threshold)
            return centers[mask]
        
        # Save original centers to CSV for debugging
        centers = np.array(centers)
        centers_orig = centers
        np.savetxt(r'C:\Users\Public\CapstoneUI\centers.csv', centers, delimiter=',', header='X,Y,Z')
        
        # Iterative filtering approach for fitting axis to centers
        for iqr_mult in [1.8, 1.2, 0.8, 1.0]:
            c_axis, axis_dir = fit_axis(centers, approx_axis)
            centers = filter_outliers(centers, c_axis, axis_dir, iqr_mult)
        
        # Final fit: Compute final axis and standard deviation
        c_axis, axis_dir = fit_axis(centers, approx_axis)
        projections = np.dot(centers - c_axis, axis_dir)
        points_on_line = c_axis + np.outer(projections, axis_dir)
        distances = np.linalg.norm(centers - points_on_line, axis=1)
        std_dev = np.std(distances)
        
        # Store results
        self.axis_loc = c_axis
        self.spindle_axis = axis_dir
        self.centers_std_dev = std_dev

        if plot_flag:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot original centers in red
            ax.scatter(centers_orig[:, 0], centers_orig[:, 1], centers_orig[:, 2], c='red', s=5, label='Original Centers')

            # Plot final centers in green
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='green', label='Final Centers')

            # Plot final axis as a black line
            # Extend axis line to span the range of data
            t = np.linspace(-np.max(np.abs(projections)), np.max(np.abs(projections)), 2)
            line_points = c_axis + np.outer(t, axis_dir)
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], c='black', label='Fitted Axis')

            # Set labels and legend
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Stddev: {self.centers_std_dev:.3f}mm. '
                         f'Using {len(centers)} of {len(centers_orig)} centers ({len(centers)/len(centers_orig)*100:.1f}%)')
            ax.legend()
            # Ensure equal aspect ratio for all axes
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for X, Y, Z
            # Compute symmetric limits for all axes
            all_points = np.vstack((centers_orig, centers, line_points))
            x_range = np.ptp(all_points[:, 0])
            y_range = np.ptp(all_points[:, 1])
            z_range = np.ptp(all_points[:, 2])
            max_range = max(x_range, y_range, z_range) / 2
            center = np.mean(all_points, axis=0)
            zoom = 8
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range/zoom, center[1] + max_range/zoom)
            ax.set_zlim(center[2] - max_range/zoom, center[2] + max_range/zoom)

            # Save and show plot
            plt.show()

        #endregion

#endregion

    def visualize_axes(self, length=0.1):
        # Red
        origin_x = o3d.geometry.LineSet()
        origin_x.points = o3d.utility.Vector3dVector([[0, 0, 0], [10, 0, 0]])
        origin_x.lines = o3d.utility.Vector2iVector([[0, 1]])
        origin_x.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

        # Green
        origin_y = o3d.geometry.LineSet()
        origin_y.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 10, 0]])
        origin_y.lines = o3d.utility.Vector2iVector([[0, 1]])
        origin_y.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

        # Blue
        origin_z = o3d.geometry.LineSet()
        origin_z.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 10]])
        origin_z.lines = o3d.utility.Vector2iVector([[0, 1]])
        origin_z.colors = o3d.utility.Vector3dVector([[0, 0, 1]])

        spindle_pcd = Numpy_to_Open3D(self.spindle_cloud)
        bar_pcd = Numpy_to_Open3D(self.bar_faces)

        # Spindle Axis
        spindle_points = np.array([self.axis_loc - length * self.spindle_axis, self.axis_loc + length * self.spindle_axis])
        spindle_axis = o3d.geometry.LineSet()
        spindle_axis.points = o3d.utility.Vector3dVector(spindle_points)
        spindle_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
        spindle_axis.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

        o3d.visualization.draw_geometries([spindle_pcd, bar_pcd, origin_x, origin_y, origin_z])

    def calc_angles(self):
        B = np.array(self.bar_axis) / np.linalg.norm(self.bar_axis)
        S = np.array(self.spindle_axis) / np.linalg.norm(self.spindle_axis)
        if not np.isclose(np.linalg.norm(B), 1.0) or not np.isclose(np.linalg.norm(S), 1.0):
            raise ValueError("Inputs must be unit vectors")
        self.bar_align = np.degrees(np.array([np.arccos(B[0]), np.arccos(B[1]), np.arccos(B[2])]))
        self.spindle_align = np.degrees(np.array([np.arccos(S[0]), np.arccos(S[1]), np.arccos(S[2])]))
        R = np.cross(B, S)
        sin_theta = np.linalg.norm(R)
        theta_deg = np.degrees(np.arcsin(sin_theta))
        self.total_angle = theta_deg
        Rx_deg, Ry_deg, Rz_deg = np.degrees(R)
        self.relative_angle = np.array([Rx_deg, Ry_deg, Rz_deg])

    def calc_toe_camber(self):
        # Bar axis calculations
        v_x, v_y, v_z = self.bar_axis
        # Toe angle: angle between y-axis and projection on x-y plane
        toe = 90-np.degrees(np.arccos(v_y / np.sqrt(v_x**2 + v_y**2))) if v_x**2 + v_y**2 != 0 else 0
        self.bar_toe = toe if v_x >= 0 else -toe  # Positive v_x: toe-in, negative: toe-out
        # Camber angle: angle between z-axis and projection on x-z plane
        camber = 90-np.degrees(np.arccos(v_z / np.sqrt(v_x**2 + v_z**2))) if v_x**2 + v_z**2 != 0 else 0
        self.bar_camber = camber if v_x >= 0 else -camber  # Positive v_x: positive camber
        
        # Spindle axis calculations
        v_x, v_y, v_z = self.spindle_axis
        # Toe angle: angle between y-axis and projection on x-y plane
        toe = 90-np.degrees(np.arccos(v_y / np.sqrt(v_x**2 + v_y**2))) if v_x**2 + v_y**2 != 0 else 0
        self.spindle_toe = toe if v_x >= 0 else -toe  # Positive v_x: toe-in, negative: toe-out
        # Camber angle: angle between z-axis and projection on x-z plane
        camber = 90-np.degrees(np.arccos(v_z / np.sqrt(v_x**2 + v_z**2))) if v_x**2 + v_z**2 != 0 else 0
        self.spindle_camber = camber if v_x >= 0 else -camber  # Positive v_x: positive camber
    
        # Calculate spindle toe and camber relative to the bar
        # Normalize bar_axis to define local x-axis
        u_x = self.bar_axis / np.linalg.norm(self.bar_axis)

        # Choose reference vector for defining local y-axis
        ref = np.array([0, 0, 1])
        if np.abs(u_x[2]) > 0.99999:  # If bar is nearly vertical, use global x-axis
            ref = np.array([1, 0, 0])

        # Define local y- and z-axes
        u_y = np.cross(ref, u_x)
        u_y = u_y / np.linalg.norm(u_y)
        u_z = np.cross(u_x, u_y)

        # Form rotation matrix (columns are local axes in global frame)
        R_bar = np.column_stack((u_x, u_y, u_z))

        # Transform spindle_axis to bar's local coordinate system
        spindle_local = R_bar.T @ self.spindle_axis
        v_x_local, v_y_local, v_z_local = spindle_local

        # Compute relative toe angle (in local x-y plane)
        if v_x_local**2 + v_y_local**2 != 0:
            toe_relative = 90 - np.degrees(np.arccos(v_y_local / np.sqrt(v_x_local**2 + v_y_local**2)))
        else:
            toe_relative = 0
        
        # Compute relative camber angle (in local x-z plane)
        if v_x_local**2 + v_z_local**2 != 0:
            camber_relative = 90 - np.degrees(np.arccos(v_z_local / np.sqrt(v_x_local**2 + v_z_local**2)))
        else:
            camber_relative = 0

        if self.side == 'left':
            self.toe = -toe_relative if v_x_local >= 0 else toe_relative
            self.camber = camber_relative if v_x_local >= 0 else -camber_relative
            self.spindle_align = np.array([-self.spindle_toe, self.spindle_camber])
            self.bar_align = np.array([-self.bar_toe, self.bar_camber])
        elif self.side == 'right':
            self.toe = toe_relative if v_x_local >= 0 else -toe_relative
            self.camber = -camber_relative if v_x_local >= 0 else camber_relative
            self.spindle_align = np.array([self.spindle_toe, -self.spindle_camber])
            self.bar_align = np.array([self.bar_toe, -self.bar_camber])


        # Total misalignment: angle between spindle and bar vectors
        spindle_norm = self.spindle_axis / np.linalg.norm(self.spindle_axis)
        bar_norm = self.bar_axis / np.linalg.norm(self.bar_axis)
        dot_product = np.clip(np.dot(spindle_norm, bar_norm), -1.0, 1.0)  # Avoid numerical errors
        self.total_misalign = np.degrees(np.arccos(dot_product))

    def print_angles(self):
        np.set_printoptions(precision=6, suppress=True)
        print(f'\nBar Axis:\t{self.bar_axis}')
        print(f'Spindle Axis:\t{self.spindle_axis}')
        np.set_printoptions(precision=4, suppress=True)
        print(f'\n---DEGREES Toe/Camber---\nBar Global Alignment:\t\t{self.bar_align}')
        print(f'Spindle Global Alignment:\t{self.spindle_align}')
        print(f'Spindle/Bar:\t{self.toe:.4f}, {self.camber:.4f}')
        print(f'Total Misalignment:\t{self.total_misalign:.4f}')

    def save_angles_to_csv(self, csv_filename="angles_output.csv", note="Alignment data for spindle and bar"):
        import os
        import pandas as pd
        """
        Appends bar and spindle axis, alignment, relative toe/camber, and total misalignment to a CSV file.
        Places note above headers and updates average and standard deviation rows.

        Args:
            csv_filename (str): Output CSV file name.
            note (str): Note to include above the header row.
        """
        # Define column headers
        headers = ['file',
            'bar_axis_x', 'bar_axis_y', 'bar_axis_z',
            'spindle_axis_x', 'spindle_axis_y', 'spindle_axis_z',
            'bar_toe', 'bar_camber',
            'spindle_toe', 'spindle_camber',
            'relative_toe', 'relative_camber',
            'total_misalign'
        ]
        dtypes = {
        'file': str,
        'bar_axis_x': float, 'bar_axis_y': float, 'bar_axis_z': float,
        'spindle_axis_x': float, 'spindle_axis_y': float, 'spindle_axis_z': float,
        'bar_toe': float, 'bar_camber': float,
        'spindle_toe': float, 'spindle_camber': float,
        'relative_toe': float, 'relative_camber': float,
        'total_misalign': float
}

        # Create new data row
        new_data = np.concatenate([[self.filename],
            self.bar_axis,
            self.spindle_axis,
            self.bar_align,
            self.spindle_align,
            [self.toe, self.camber, self.total_misalign]
        ])

        # Initialize DataFrame for new data
        new_df = pd.DataFrame([new_data], columns=headers).astype(dtypes)

        # Read existing CSV if it exists
        if os.path.exists(csv_filename):
            # Read, skipping note and header rows
            existing_df = pd.read_csv(csv_filename, skiprows=2, header=None, names=headers, dtype=dtypes)
            # Filter out 'avg' and 'stddev' rows
            data_rows = existing_df[
                ~existing_df['file'].isin(['avg', 'stddev'])
            ]
            # Append new data
            data_rows = pd.concat([data_rows, new_df], ignore_index=True)
        else:
            data_rows = new_df

        # Calculate average and standard deviation (excluding 'file' column)
        avg_row = data_rows.drop(columns=['file']).mean().to_numpy()
        std_row = data_rows.drop(columns=['file']).std().to_numpy()

        # Create note row with proper column alignment
        note_row = pd.DataFrame([[note] + [''] * (len(headers) - 1)], columns=headers)

        # Create header row
        header_row = pd.DataFrame([headers], columns=headers)

        # Create data DataFrame
        data_df = data_rows.copy()

        # Create average and standard deviation rows
        avg_df = pd.DataFrame([['avg'] + avg_row.tolist()], columns=headers)
        std_df = pd.DataFrame([['stddev'] + std_row.tolist()], columns=headers)

        # Combine all rows: note, headers, data, average, std
        final_df = pd.concat([note_row, header_row, data_df, avg_df, std_df], ignore_index=True)

        # Save to CSV without adding an extra header
        final_df.to_csv(csv_filename, index=False, float_format='%.6f', header=False)

    def plot_vectors(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Origin
        origin = np.array([0, 0, 0])
        
        # Plot vectors
        ax.quiver(*origin, *self.bar_axis, color='b', label='bar')
        ax.quiver(*origin, *self.spindle_axis, color='r', label='spindle')
        ax.quiver(*origin, 1, 0, 0, color='k', linestyle='--', label='X-axis')
        ax.quiver(*origin, 0, 1, 0, color='g', linestyle='--', label='Y-axis')
        
        # Set equal aspect ratio
        max_range = max(np.linalg.norm(self.bar_axis), np.linalg.norm(self.spindle_axis)) * 1.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

def Trim_Cloud(cloud, direction, cutOff=[-500,500], minPoints=10000):
    untrimmed_cloud = cloud
    if direction == 'x':
        index = 0
    elif direction == 'y':
        index = 1
    elif direction == 'z':
        index = 2
    else:
        print(f'Trimming Error: Invalid direction slection, returning untrimmed cloud')
        return untrimmed_cloud

    valid_mask = (cloud[index] >= cutOff[0]) & (cloud[index] <= cutOff[1])
    cloud = cloud[:, valid_mask]
    if cloud.shape[1] >= minPoints:
        return cloud
    else:
        print(f'Trimming Error: {cloud.shape[1]} points less than {minPoints} points in trimmed cloud. Returning untrimmed cloud')

def Numpy_to_Open3D(cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.T)
    return pcd

def Open3D_to_Numpy(pcd):
    return np.asarray(pcd.points).T

def visualize_axis(pcd, c_axis, axis_dir, length=0.1):
    axis_points = np.array([c_axis - length * axis_dir, c_axis + length * axis_dir])
    axis_line = o3d.geometry.LineSet()
    axis_line.points = o3d.utility.Vector3dVector(axis_points)
    axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    axis_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    o3d.visualization.draw_geometries([pcd, axis_line])

def Rotate(threeD_Object, axis, angle):
    angle = np.radians(angle)
    if axis == 'x':
        R = np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])    
    elif axis == 'y':
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    return np.dot(R, threeD_Object)

def Rotation_to_Zaxis(normal):
    normal = np.array(normal[:3])
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    cos_theta = np.dot(normal, z_axis)
    sin_theta = np.linalg.norm(v)
    if np.isclose(sin_theta, 0):
        return np.eye(3) if cos_theta > 0 else -np.eye(3)
    v_x, v_y, v_z = v
    K = np.array([[0, -v_z, v_y],
                  [v_z, 0, -v_x],
                  [-v_y, v_x, 0]])
    R = np.eye(3) + K + (1 - cos_theta)*np.dot(K, K)/sin_theta
    i = 0
    while True:
        R = enforce_rotation_properties(R)
        if np.allclose(np.dot(R.T, R), np.eye(3)) and np.isclose(np.linalg.det(R), 1.0):
            return R
        if i > 10:
            return R
        i += 1

def make_orthogonal(matrix):
    U, _, Vt = np.linalg.svd(matrix)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    return R

def enforce_rotation_properties(matrix):
    if not np.allclose(np.dot(matrix.T, matrix), np.eye(3), atol=1e-6):
        matrix = make_orthogonal(matrix)
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        U, _, Vt = np.linalg.svd(matrix)
        matrix = np.dot(U, Vt)
    return matrix

def Check_Scaling(base_cloud, transformed_cloud, ui=None):
    base_cloud = base_cloud - np.mean(base_cloud, axis=1)[:, np.newaxis]
    transformed_cloud = transformed_cloud - np.mean(transformed_cloud, axis=1)[:, np.newaxis]
    original_norm = np.linalg.norm(base_cloud, axis=0).mean()
    transformed_norm = np.linalg.norm(transformed_cloud, axis=0).mean()
    if not np.isclose(original_norm, transformed_norm):
        if ui:
            ui.log_message("Scaling distortion detected")
        else:
            print("Scaling distortion detected")

def Normal_of_Rotated_Plane(axis, angle):
    angle_rad = np.radians(angle)
    if axis == 'x':
        initial_normal = np.array([0, 0, 1])
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle_rad), -np.sin(angle_rad)],
                      [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        initial_normal = np.array([0, 0, 1])
        R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        initial_normal = np.array([1, 0, 0])
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                      [np.sin(angle_rad), np.cos(angle_rad), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    rotated_normal = np.dot(R, np.array(initial_normal))
    normalized_normal = rotated_normal / np.linalg.norm(rotated_normal)
    normalized_normal[np.abs(normalized_normal) < 1e-12] = 0.0
    return normalized_normal

def Calc_Plane(points, title=0, plotNum=0, numPoints=1000, ui=None):
    # Shrinks points to fewer points
    sampled_indices = np.linspace(0, points.shape[1] - 1, numPoints, dtype=int)
    sampled_points = points[:, sampled_indices]
    if ui:
        ui.log_message(f'Fitting plane to {sampled_points.shape[1]} points')
    else:
        print(f'Fitting plane to {sampled_points.shape[1]} points')

    def fit_plane(points):
        centroid = np.mean(points, axis=1, keepdims=True)
        svd = np.linalg.svd(points - centroid)
        normal_vector = svd[0][:, -1]
        d = -normal_vector.dot(centroid.flatten())
        plane = np.hstack((normal_vector, d))
        return plane, centroid.flatten()

    def filter_plane(points, normal_vector, iqr_scale):
        d = -normal_vector.dot(centroid.flatten())
        ortho_dist = np.abs(normal_vector[0] * points[0] +
                            normal_vector[1] * points[1] +
                            normal_vector[2] * points[2] + d) / np.linalg.norm(normal_vector)
        Q1, Q3 = np.percentile(ortho_dist, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_scale * IQR
        upper_bound = Q3 + iqr_scale * IQR
        filtered_points = points[:, (ortho_dist >= lower_bound) & (ortho_dist <= upper_bound)]
        return filtered_points

    filt_points = sampled_points
    for i, iqr_scale in enumerate([1.0, 0.5]):
        if ui:
            ui.log_message(f"\tIteration {i}: filtering {filt_points.shape[1]} points")
        else:
            print(f"\tIteration {i}: filtering {filt_points.shape[1]} points")
        plane, centroid = fit_plane(filt_points)
        filt_points = filter_plane(filt_points, plane[0:3], iqr_scale)
    final_filt_points = filt_points
    if ui:
        ui.log_message(f"\tIteration {i+1}: final {final_filt_points.shape[1]} points")
    else:
        print(f"\tIteration {i+1}: final {final_filt_points.shape[1]} points")
    plane_final, centroid_final = fit_plane(final_filt_points)
    normal_final = plane_final[0:3]
    d_final = -normal_final.dot(centroid_final.flatten())
    
    x_angle = np.degrees(np.arccos(abs(normal_final[0]) / np.linalg.norm(normal_final)))
    y_angle = np.degrees(np.arccos(abs(normal_final[1]) / np.linalg.norm(normal_final)))
    z_angle = np.degrees(np.arccos(abs(normal_final[2]) / np.linalg.norm(normal_final)))

    ortho_dist_final = np.abs(normal_final[0] * final_filt_points[0] +
                              normal_final[1] * final_filt_points[1] +
                              normal_final[2] * final_filt_points[2] + d_final) / np.linalg.norm(normal_final)
    uncertainty_95 = np.std(ortho_dist_final) * 1.96
    
    if plotNum != 0:
        fig = plt.figure(plotNum)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sampled_points[0], sampled_points[1], sampled_points[2], color='red', s=1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Best Fit Plane w/Filtering')
        ax.scatter(final_filt_points[0], final_filt_points[1], final_filt_points[2], color='green', s=10)
        xx, yy = np.meshgrid(np.linspace(np.min(final_filt_points[0]), np.max(final_filt_points[0]), 10),
                            np.linspace(np.min(final_filt_points[1]), np.max(final_filt_points[1]), 10))
        zz = (-normal_final[0] * xx - normal_final[1] * yy - d_final) / normal_final[2]
        ax.plot_surface(xx, yy, zz, alpha=0.5, color='yellow')
        if title:
            ax.set_title(title)
            fig.suptitle(f'x: {90 - x_angle:.3f}, y: {90 - y_angle:.3f}, z: {z_angle:.3f}')
        plt.figure(10+plotNum)
        plt.hist(ortho_dist_final, bins=17, density=True)
        plt.axvline(uncertainty_95, c='red')
        plt.title(f'Distribution of Plane Error (95% Spread: ±{uncertainty_95:.2f} mm)')
        plt.xlabel('Orthogonal Distance - Point to Plane (mm)')
        plt.show()

    return plane_final, np.array([x_angle, y_angle, z_angle]), uncertainty_95

def Plot_Cloud_PyVista(points, pointSize=1.0, ui=None):
    cloud = pv.PolyData(points.T)
    if ui:
        ui.log_message(f'Displaying {points.shape[1]} points in PyVista window\n\tTo continue, select PyVista window and press q\n\tDo not close PyVista window')
    else:
        print(f'Displaying {points.shape[1]} points')
    time.sleep(0.1)
    z_values = points[2]
    cloud.point_data['z'] = z_values
    cloud.point_data.set_array(z_values, 'z')
    plotter = pv.Plotter()
    plotter.set_background('gray')
    plotter.add_mesh(cloud, scalars='z', cmap='coolwarm', point_size=pointSize)
    plotter.add_axes(xlabel='X (Axle)', ylabel='Y (Motion)', zlabel='Z (Vertical)',
                     line_width=4, labels_off=False)
    
    def orbit_left():
        plotter.camera.azimuth -= 1
        plotter.render()

    def orbit_right():
        plotter.camera.azimuth += 1
        plotter.render()

    def orbit_up():
        plotter.camera.elevation += 1
        plotter.render()

    def orbit_down():
        plotter.camera.elevation -= 1
        plotter.render()

    def save_image():
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
        filename = f"point_cloud_{timestamp}.png"
        plotter.screenshot(filename)
        if ui:
            ui.log_message(f"Image saved as {filename}")
        else:
            print(f"Image saved as {filename}")

    plotter.add_key_event('4', orbit_left)
    plotter.add_key_event('6', orbit_right)
    plotter.add_key_event('8', orbit_up)
    plotter.add_key_event('2', orbit_down)
    plotter.add_key_event('s', save_image)
    plotter.camera_position = 'xy'
    plotter.camera_set = True
    plotter.reset_camera()
    plotter.camera.SetParallelProjection(True)
    plotter.show(auto_close=False, interactive=True)

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY

def Find_Ledges_Along_Normal(cloud, normal, ledgeThreshold, shortLedge, closeLedges, ui=None):
    if ui:
        ui.log_message('Grouping filtered surfaces into ledges')
    else:
        print('Grouping filtered surfaces into ledges')
    normal = normal / np.linalg.norm(normal)
    proj_dist = np.dot(np.vstack(cloud).T, normal)
    sorted_indices = np.argsort(proj_dist)
    sorted_proj_dist = proj_dist[sorted_indices]
    ledges = []
    current_ledge = [sorted_indices[0]]
    for i in range(1, len(sorted_indices)):
        if sorted_proj_dist[i] - sorted_proj_dist[i-1] < ledgeThreshold:
            current_ledge.append(sorted_indices[i])
        else:
            ledges.append(np.array(current_ledge))
            current_ledge = [sorted_indices[i]]
    if current_ledge:
        ledges.append(np.array(current_ledge))
    ledge_avgs = []
    for ledge in ledges:
        ledge_points = np.vstack([cloud[0][ledge], cloud[1][ledge], cloud[2][ledge]]).T
        avg = np.mean(np.dot(ledge_points, normal))
        ledge_avgs.append(avg)
    sorted_ledge_indices = np.argsort(ledge_avgs)
    sorted_ledges = [ledges[i] for i in sorted_ledge_indices]
    sorted_ledge_avgs = [ledge_avgs[i] for i in sorted_ledge_indices]
    uf = UnionFind(len(sorted_ledges))
    for i in range(len(sorted_ledges) - 1):
        if sorted_ledge_avgs[i + 1] - sorted_ledge_avgs[i] < closeLedges:
            uf.union(i, i + 1)
    merged_ledges = []
    merged_ledge_avgs = []
    current_root = uf.find(0)
    current_group = [sorted_ledges[0]]
    for i in range(1, len(sorted_ledges)):
        if uf.find(i) == current_root:
            current_group.append(sorted_ledges[i])
        else:
            combined_ledge = np.concatenate(current_group)
            ledge_points = np.vstack([cloud[0][combined_ledge], cloud[1][combined_ledge], cloud[2][combined_ledge]]).T
            avg = np.mean(np.dot(ledge_points, normal))
            merged_ledges.append(combined_ledge)
            merged_ledge_avgs.append(avg)
            current_root = uf.find(i)
            current_group = [sorted_ledges[i]]
    if current_group:
        combined_ledge = np.concatenate(current_group)
        ledge_points = np.vstack([cloud[0][combined_ledge], cloud[1][combined_ledge], cloud[2][combined_ledge]]).T
        avg = np.mean(np.dot(ledge_points, normal))
        merged_ledges.append(combined_ledge)
        merged_ledge_avgs.append(avg)
    total_points = len(cloud[0])
    big_ledges = []
    big_ledge_avgs = []
    for ledge, avg in zip(merged_ledges, merged_ledge_avgs):
        if len(ledge) > total_points * shortLedge:
            big_ledges.append(np.vstack([cloud[0][ledge], cloud[1][ledge], cloud[2][ledge]]))
            big_ledge_avgs.append(avg)
    return big_ledges, big_ledge_avgs

def Sort_Ledges(ledges, ledgeAvgs, sortType='location'):
    if sortType == 'location':
        sortedIndices = np.argsort(ledgeAvgs)
    elif sortType == 'size':
        sortedIndices = np.argsort([ledge.shape[1] for ledge in ledges])

    sortedLedges = [ledges[i] for i in sortedIndices]
    sortedLedgeAvgs = [ledgeAvgs[i] for i in sortedIndices]
    return sortedLedges, sortedLedgeAvgs

def Clean_Bar_Face(face, radius):
    centroid_xy = [np.mean(face[0]), np.mean(face[1])]
    distances = np.sqrt((face[0] - centroid_xy[0])**2 + (face[1] - centroid_xy[1])**2)
    mask = distances <= radius
    return face[:, mask]

def Bound_Spindle_2D(plane_points, show=False):
    # Define quadrant clustering function
    def cluster_quadrants(points, min_points=100, max_levels=5, tolerance=1e1):
        def divide_quadrants(points, u_mean, v_mean, level=0, prefix=''):
            if len(points) < min_points or level >= max_levels:
                return {}
            
            quadrants = {
                f'{prefix}Q1': {'points': [], 'center': None},  # u > u_mean, v > v_mean
                f'{prefix}Q2': {'points': [], 'center': None},  # u < u_mean, v > v_mean
                f'{prefix}Q3': {'points': [], 'center': None},  # u < u_mean, v < v_mean
                f'{prefix}Q4': {'points': [], 'center': None}   # u > u_mean, v < v_mean
            }
            
            # Assign points to quadrants
            for point in points:
                u, v = point
                if u > u_mean and v > v_mean:
                    quadrants[f'{prefix}Q1']['points'].append(point)
                elif u < u_mean and v > v_mean:
                    quadrants[f'{prefix}Q2']['points'].append(point)
                elif u < u_mean and v < v_mean:
                    quadrants[f'{prefix}Q3']['points'].append(point)
                elif u > u_mean and v < v_mean:
                    quadrants[f'{prefix}Q4']['points'].append(point)
            
            results = {}
            for q in quadrants:
                points_q = np.array(quadrants[q]['points'])
                if len(points_q) >= min_points:
                    center = np.mean(points_q, axis=0)
                    quadrants[q]['center'] = center
                    # Recursively divide quadrants
                    sub_results = divide_quadrants(points_q, center[0], center[1], level + 1, f'{q}_')
                    # Check if sub-quadrant center aligns with current center
                    for sub_q, sub_data in sub_results.items():
                        if sub_data['center'] is not None and np.allclose(sub_data['center'], center, atol=tolerance):
                            continue
                        results[sub_q] = sub_data
                    results[q] = quadrants[q]
            
            return results

        u_mean, v_mean = np.mean(points[:, 0]), np.mean(points[:, 1])
        return divide_quadrants(points, u_mean, v_mean)

    # Apply quadrant clustering
    min_points = 100
    clusters = cluster_quadrants(plane_points, min_points=min_points)

    # Optional: Plot clusters for visualization
    for q, data in clusters.items():
        points = np.array(data['points'])
        if len(points) >= min_points:
            plt.scatter(points[:, 0], points[:, 1], s=1, label=q)
            if data['center'] is not None:
                plt.scatter(data['center'][0], data['center'][1], c='black', marker='x')
    # Extract cluster centers
    cluster_centers = {q: data['center'] for q, data in clusters.items() if data['center'] is not None}
    # Combine cluster centers within 50 units
    combined_clusters = []
    processed = set()
    threshold = 50

    for q1, c1 in cluster_centers.items():
        if q1 in processed:
            continue
        combined_points = clusters[q1]['points']
        for q2, c2 in cluster_centers.items():
            if q2 != q1 and q2 not in processed:
                if np.linalg.norm(c1 - c2) < threshold:
                    combined_points.extend(clusters[q2]['points'])
                    processed.add(q2)
        combined_clusters.append(np.array(combined_points))
        processed.add(q1)

    if show:
        for cluster in combined_clusters:
            if len(cluster) > 0:
                center = np.mean(cluster, axis=0)
                # plt.scatter(center[0], center[1], c='red', s=50)
        plt.xlabel("u-axis")
        plt.ylabel("v-axis")
        plt.axis('equal')
        plt.legend()
        # plt.show()

    plane_spindle = max(combined_clusters, key=len)
    return plane_spindle

def Cloud_Expected_Normal_Filter(cloud, expected_normal, angle_threshold=10, ui=None):
    if ui:
        ui.log_message('Filtering by surface normals')
    else:
        print('Filtering by surface normals')
    pcd = Numpy_to_Open3D(cloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)
    cos_theta = np.dot(normals, expected_normal) / (np.linalg.norm(normals, axis=1) * np.linalg.norm(expected_normal))
    angles = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    valid_indices = np.where(angles < angle_threshold)[0]
    return Open3D_to_Numpy(pcd.select_by_index(valid_indices))

def RANSAC_Broadface_Plane(cloud, expected_normal, max_angle_deviation=10, distance_threshold=1.0, ui=None):
    pcd = Numpy_to_Open3D(cloud)
    max_angle_rad = np.radians(max_angle_deviation)
    for _ in range(1000):
        plane_model, _ = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=10)
        normal = np.array(plane_model[:3])
        dot_product = np.dot(normal, expected_normal) / (np.linalg.norm(normal) * np.linalg.norm(expected_normal))
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        if angle < max_angle_rad:
            normal[np.abs(normal) < 1e-11] = 0.0
            plane_model[np.abs(plane_model) < 1e-11] = 0.0
            return normal, np.array(plane_model)
    if ui:
        ui.log_message("Warning: No valid plane found within angle constraint.")
    else:
        print("Warning: No valid plane found within angle constraint.")
    return None, None

def RANSAC_ICP_Best_Fit(adjCloud, baseCloud, threshold=10.0, max_iteration=50):
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(adjCloud.T)
    source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(baseCloud.T)
    target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if not isinstance(source_cloud, o3d.geometry.PointCloud) or not isinstance(target_cloud, o3d.geometry.PointCloud):
        raise ValueError("Clouds must be Open3D PointCloud objects")
    initial_transform = np.eye(4)
    reg_result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    return source_cloud.transform(reg_result.transformation), reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse

def Broadface_to_Z(cloud, expected_normal, degTolerance=10, ui=None):
    if ui:
        ui.log_message("\nExpected Broadface Normal: " + str(expected_normal))
    else:
        print("\nExpected Broadface Normal: ", expected_normal)
    broadface_surfaces = Cloud_Expected_Normal_Filter(cloud, expected_normal, angle_threshold=degTolerance, ui=ui)
    broadface_ledges, _ = Find_Ledges_Along_Normal(broadface_surfaces, normal=expected_normal, ledgeThreshold=1, shortLedge=0.01, closeLedges=10, ui=ui)
    broadface_surfaces = np.hstack(broadface_ledges)
    lowest_broadface_ledge = broadface_ledges[0]
    lowest_broadface_plane, _, _ = Calc_Plane(broadface_ledges[0], ui=ui)
    bracket_normal = lowest_broadface_plane[:3]
    if ui:
        ui.log_message("Broadface Plane: " + str(lowest_broadface_plane))
    else:
        print("Broadface Plane: ", lowest_broadface_plane)
    cloud_broadface = np.copy(lowest_broadface_ledge)
    cloud_Rs_to_Z = []
    i = 0
    while True:
        R_to_Z = Rotation_to_Zaxis(bracket_normal)
        cloud_Rs_to_Z.append(R_to_Z)   
        bracket_normal = np.dot(R_to_Z, bracket_normal)
        cloud_broadface = np.dot(R_to_Z, cloud_broadface)
        skew = abs(bracket_normal[0] + bracket_normal[1])
        Check_Scaling(lowest_broadface_ledge, cloud_broadface, ui=ui)
        i += 1
        if skew < 1e-6 or i >= 10:
            break
    total_cloud_R_to_Z = reduce(np.matmul, reversed(cloud_Rs_to_Z))
    return np.dot(total_cloud_R_to_Z, cloud), total_cloud_R_to_Z

def Check_Not_Casting(ledge, ui=None):
    if ui:
        ui.log_message('Testing if bottom ledge is casting or hubface')
    else:
        print('Testing if bottom ledge is casting or hubface')
    x, y, z = ledge[0], ledge[1], ledge[2]
    center_x, center_y = np.mean(x), np.mean(y)
    radii = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    Q1 = np.percentile(radii, 25)
    Q3 = np.percentile(radii, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.0 * IQR
    upper_bound = Q3 + 1.0 * IQR
    valid_indices = (radii >= lower_bound) & (radii <= upper_bound)
    x, y, z, radii = x[valid_indices], y[valid_indices], z[valid_indices], radii[valid_indices]
    max_radius = np.max(radii)
    min_radius = np.min(radii)
    valid_indices_outer = radii >= 0.8 * max_radius
    valid_indices_inner = radii <= 1.3 * min_radius
    x_outer, y_outer, z_outer = x[valid_indices_outer], y[valid_indices_outer], z[valid_indices_outer]
    x_inner, y_inner, z_inner = x[valid_indices_inner], y[valid_indices_inner], z[valid_indices_inner]
    angles_outer = np.arctan2(y_outer - center_y, x_outer - center_x)
    angles_inner = np.arctan2(y_inner - center_y, x_inner - center_x)
    angles_outer = np.mod(angles_outer, 2*np.pi)
    angles_inner = np.mod(angles_inner, 2*np.pi)
    sorted_indices_outer = np.argsort(angles_outer)
    sorted_indices_inner = np.argsort(angles_inner)
    sorted_angles_outer = angles_outer[sorted_indices_outer]
    sorted_angles_inner = angles_inner[sorted_indices_inner]
    angular_gaps_outer = np.diff(sorted_angles_outer, append=sorted_angles_outer[0] + 2*np.pi)
    angular_gaps_inner = np.diff(sorted_angles_inner, append=sorted_angles_inner[0] + 2*np.pi)
    gap_threshold = 2*np.pi/15
    large_gaps_outer = np.sum(angular_gaps_outer > gap_threshold)
    large_gaps_inner = np.sum(angular_gaps_inner > gap_threshold)
    if ui:
        ui.log_message(f'Large angular gaps (wheel mounts) on outer,inner perimeter: {large_gaps_outer},{large_gaps_inner}')
    else:
        print(f'Large angular gaps (wheel mounts) on outer,inner perimeter: {large_gaps_outer},{large_gaps_inner}')
    if large_gaps_inner < 4 or large_gaps_outer >= 5:
        return True
    else:
        return False

def Find_HubFace(ledges, ledgeAvgs, reverse=False, deleteGround=False, ui=None):
    ledges = deepcopy(ledges)
    ledgeAvgs = ledgeAvgs[:]
    ledgeNumPoints = [len(ledge[0]) for ledge in ledges]
    descendIndices_numPoints = np.argsort(ledgeNumPoints)[::-1]
    biggestLedge_numPoints = len(ledges[descendIndices_numPoints[0]][0])
    sortedIndices = np.argsort(ledgeAvgs)[::-1] if reverse else np.argsort(ledgeAvgs)
    while len(ledges[sortedIndices[0]][0]) / biggestLedge_numPoints <= 0.50:
        del ledges[sortedIndices[0]]
        del ledgeAvgs[sortedIndices[0]]
        del ledgeNumPoints[sortedIndices[0]]
        sortedIndices = np.argsort(ledgeAvgs)[::-1] if reverse else np.argsort(ledgeAvgs)
        ledgeNumPoints = [len(ledge[0]) for ledge in ledges]
        descendIndices_numPoints = np.argsort(ledgeNumPoints)[::-1]
        biggestLedge_numPoints = len(ledges[descendIndices_numPoints[0]][0])
    if deleteGround:
        if ui:
            ui.log_message('Ignoring Ground in Find_HubFace()')
        else:
            print('Ignoring Ground in Find_HubFace()')
        del ledges[sortedIndices[0]]
        del ledgeAvgs[sortedIndices[0]]
        del ledgeNumPoints[sortedIndices[0]]
        sortedIndices = np.argsort(ledgeAvgs)[::-1] if reverse else np.argsort(ledgeAvgs)
        ledgeNumPoints = [len(ledge[0]) for ledge in ledges]
        descendIndices_numPoints = np.argsort(ledgeNumPoints)[::-1]
        biggestLedge_numPoints = len(ledges[descendIndices_numPoints[0]][0])
    while len(ledges[sortedIndices[0]][0]) / biggestLedge_numPoints <= 0.50:
        del ledges[sortedIndices[0]]
        del ledgeAvgs[sortedIndices[0]]
        del ledgeNumPoints[sortedIndices[0]]
        sortedIndices = np.argsort(ledgeAvgs)[::-1] if reverse else np.argsort(ledgeAvgs)
        ledgeNumPoints = [len(ledge[0]) for ledge in ledges]
        descendIndices_numPoints = np.argsort(ledgeNumPoints)[::-1]
        biggestLedge_numPoints = len(ledges[descendIndices_numPoints[0]][0])
    numPoints0 = len(ledges[sortedIndices[0]][0])
    numPoints1 = len(ledges[sortedIndices[1]][0])
    if numPoints1 / numPoints0 >= 0.15:
        if ui:
            ui.log_message('Checking if hubface spread is OK')
        else:
            print('Checking if hubface spread is OK')
        _, _, spread = Calc_Plane(ledges[sortedIndices[1]], numPoints=5000, ui=ui)
        if ui:
            ui.log_message(str(spread))
        else:
            print(spread)
        if spread <= 0.15:
            hubFace = ledges[sortedIndices[1]]              
            hubFaceAvg = ledgeAvgs[sortedIndices[1]]
        else:
            hubFace = ledges[sortedIndices[2]]              
            hubFaceAvg = ledgeAvgs[sortedIndices[2]]
    else:
        if Check_Not_Casting(ledges[sortedIndices[0]], ui=ui):
            hubFace = ledges[sortedIndices[0]]
            hubFaceAvg = ledgeAvgs[sortedIndices[0]]
        else:
            if ui:
                ui.log_message('Checking if hubface spread is OK')
            else:
                print('Checking if hubface spread is OK')
            _, _, spread = Calc_Plane(ledges[sortedIndices[1]], numPoints=5000, ui=ui)
            if ui:
                ui.log_message(str(spread))
            else:
                print(spread)
            if spread <= 0.15:
                hubFace = ledges[sortedIndices[1]]          
                hubFaceAvg = ledgeAvgs[sortedIndices[1]]
            else:
                hubFace = ledges[sortedIndices[2]]              
                hubFaceAvg = ledgeAvgs[sortedIndices[2]]
    return hubFace, hubFaceAvg
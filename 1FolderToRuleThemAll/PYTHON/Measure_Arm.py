import numpy as np
from TorFlex_Alignment import Torsion_Arm_LJS640
import time

def main(filename=None, auto_flag=False, scan_type='live', ui=None, debug_flag=False):
    start = time.time()
    if filename == None:
        filename = r'C:\Users\Public\CapstoneUI\temporary_scan.csv'

    # Load arm scan
    index = 3
    scan1 = Torsion_Arm_LJS640(filename, view_angle_horizontal=45, scanType=scan_type,
                               cutOff=[-500, 500, -500, 200, -500, 500], # x, y, z min & max
                               ui=ui)
    # scan1.show_cloud()

    # Prepare data
    scan1.center_cloud()
    # scan1.rotate_cloud(axis='z', angle=180)
    if debug_flag:
        scan1.show_cloud()

    # Find bar faces
    scan1.fit_bar_faces(plotNum=0, cutOff=[0, 200], show=debug_flag)    #-25, 300
    # print(f'Bar Axis: {scan1.bar_axis}')

    # Find spindle
    #scan1.fit_spindle(axial_cutoff=-100, num_bins=100, circle_fit_tol=0.18, show=True, plot=False)
    #scan1.fit_spindle2(axial_cutoff=-80, num_bins=80, circle_fit_tol=0.2, circle_resid_tol=[1.0], min_fit_points=300, centers_resid_tol=[2.0], show=False, plot=False)
    # print(f'Spindle Axis: {scan1.axis_dir}')
    scan1.fit_spindle_3D3(axial_cutoff=-150, side='left', show_flag=debug_flag, plot_flag=False, box_size=8.0)     #-70
    # print(f'Spindle Axis: {scan1.spindle_axis}')

    # scan1.visualize_axes(length=100)

    # scan1.calc_angles()
    scan1.calc_toe_camber(side='left')
    scan1.print_angles()    
    scan1.save_angles_to_csv()
    # scan1.plot_vectors()

    end = time.time()

    # scan1.show_cloud(np.hstack((scan1.barPrimaryFace, scan1.barSecondaryFace, scan1.spindle_cloud)))

    print(f'\nTotal Duration: {end-start:.3f}')

    results = {"bar_toe": scan1.bar_align[0], "bar_camber": scan1.bar_align[1], 
               "spindle_toe": scan1.spindle_align[0], "spindle_camber": scan1.spindle_align[1],
               "toe": scan1.toe, "camber": scan1.camber, 
               "total_misalign": scan1.total_misalign}
    return results

if __name__ == "__main__":
    main(filename=r'RealScans\Perfect Arm\Perfect_ArmA10.csv', scan_type='live', debug_flag=True)
    # main(filename=r'3D Simulation\SimScans\CrankArm22.5-right_0toe_-5cam..txt', scan_type='sim')
 
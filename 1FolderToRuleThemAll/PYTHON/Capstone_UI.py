import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
import time
import os
import threading
from datetime import date
from reportlab.pdfgen import canvas
import webbrowser
import Measure_Hub as MH  
import Measure_Arm as MA
import Perform_Scan_LJS640 as PS

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class Dexter_Capstone_UI:
    
    # Setup
    def __init__(self, master):
        """Initialize the UI with window settings and file paths."""
        self.master = master
        self.master.title("Dexter-Capstone 2025")
        self.master.geometry("1200x800")
        self.default_button_size = 300
        self.axle_database_path = r"C:\Users\Public\CapstoneUI\Axle_Database.csv"
        self.arm_database_path = r"C:\Users\Public\CapstoneUI\Arm_Database.csv"
        self.calibration_path = r"C:\Users\Public\CapstoneUI\Calibration History.csv"
        self.temp_scan_pathA = r'C:\Users\Public\CapstoneUI\temporary_scan.csv'
        self.get_hub_calibration()
        
        # Create a persistent frame for the message log at the bottom
        self.log_frame = ctk.CTkFrame(self.master)
        self.log_frame.pack(side=ctk.BOTTOM, fill=ctk.X, padx=20, pady=(0, 20))
        self.message_log = ctk.CTkTextbox(self.log_frame, height=150, state='disabled')
        self.message_log.pack(fill=ctk.BOTH, expand=True)
        
        # Create a main content frame above the log
        self.content_frame = ctk.CTkFrame(self.master)
        self.content_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=(20, 0))
        
        self.open_home_screen()

    def clear_window(self):
        """Remove all widgets from the content frame and unbind the Return key."""
        self.master.unbind("<Return>")
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def setup_screen(self, title_text, content_callback, home_button=True):
        """Set up a screen with a title, content, and optional home button, preserving the message log."""
        self.clear_window()
        
        # Add title
        title = ctk.CTkLabel(self.content_frame, text=title_text, font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=(20, 40))
        
        # Call the content callback to add specific content
        content_callback(self.content_frame)
        
        # Add home button if requested
        if home_button:
            bottom_frame = ctk.CTkFrame(self.content_frame)
            bottom_frame.pack(side=ctk.BOTTOM, fill=ctk.X, pady=(20, 0))
            ctk.CTkButton(master=bottom_frame, text="Return to Main Menu", command=self.open_home_screen, width=self.default_button_size).pack()

    def log_message(self, message):
        """Append a message to the log and scroll to the bottom."""
        def update_log():
            if hasattr(self, 'message_log') and self.message_log:
                self.message_log.configure(state='normal')
                self.message_log.insert(ctk.END, message + '\n')
                self.message_log.configure(state='disabled')
                self.message_log.see(ctk.END)
            else:
                print(f"Fallback: {message}")
        self.master.after(0, update_log)

    def update_status(self, message):
        """Update the log with the given message (replaces old status label functionality)."""
        self.log_message(message)

    def get_input(self, message, options=["Yes", "No"]):
        """Show a prompt with buttons and return the user's choice."""
        self.user_response = None
        self.response_event = threading.Event()

        def content(frame):
            ctk.CTkLabel(frame, text=message, font=ctk.CTkFont(size=18), wraplength=1000).pack(pady=(20, 20))
            for option in options:
                ctk.CTkButton(frame, text=option, command=lambda opt=option: self.set_response(opt)).pack(pady=(10, 0))

        self.setup_screen("Manual Interaction", content, home_button=False)
        self.response_event.wait()  # Block until user responds
        return self.user_response

    def set_response(self, response):
        """Set the user's response and signal completion."""
        self.user_response = response
        self.response_event.set()


    # General
    def run_scanner(self):
        def content(frame):
            ctk.CTkLabel(frame, text="Scanning...", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
        if self.type == 'hub':
            self.setup_screen("TorFlex Axle — Measure Hub Alignment", content, home_button=False)
        elif self.type == 'arm':
            self.setup_screen("TorFlex Axle — Measure Arm Alignment", content, home_button=False)
        self.master.update()

        data = PS.perform_scan().astype(float)
        for i in data:
            i = (i - 2**15) * .0102
        np.savetxt(self.temp_scan_pathA, data, delimiter=',', header='X Y Z')
        
        self.scan_type = 'live'
        if self.type == 'hub':
            self.hub_scan_fileA = self.temp_scan_pathA
            self.calc_hub_alignment()
        elif self.type =='arm':
            self.arm_scan_fileA = self.temp_scan_pathA
            self.calc_arm_alignment()
    
    def validate_file_and_start(self):
        scan_file = self.existing_scan_entry.get().strip()
        self.scan_type = 'real'
        if not scan_file or not os.path.isfile(scan_file):
            messagebox.showerror("Error", "Please enter a valid and accessible file path.")
            return
        if self.type == 'hub':
            self.hub_scan_fileA = scan_file
            self.calc_hub_alignment()
        elif self.type == 'arm':
            self.arm_scan_fileA = scan_file
            self.calc_arm_alignment()


    # Home screen
    def open_home_screen(self):
        """Display the home screen with navigation options."""
        def content(frame):
            ctk.CTkButton(frame, text="Measure Hub", command=self.measure_hub, width=self.default_button_size).pack(pady=(0, 20))
            ctk.CTkButton(frame, text="Measure Crank Arm", command=self.measure_arm, width=self.default_button_size).pack(pady=(0, 20))
            ctk.CTkButton(frame, text="Hub Measurement Calibration", command=self.calibrate_hub, width=self.default_button_size).pack(pady=(0, 20))
            ctk.CTkButton(frame, text="Crank Arm Measurement Calibration", command=self.calibrate_arm, width=self.default_button_size).pack(pady=(0, 20))

        self.setup_screen("Dexter TorFlex Axle — Hub and Crank Arm Alignment Software", content, home_button=False)

    def measure_hub(self):
        self.type = 'hub'
        """Show screen to enter axle ID for hub measurement."""
        def save_axleID():
            self.axle_id = self.barcode_entry.get().strip()
            if not self.axle_id:
                messagebox.showerror("Error", "Please enter an Axle ID")
                return

            os.makedirs(os.path.dirname(self.axle_database_path), exist_ok=True)
            self.initialize_csv(self.axle_database_path, ["Axle ID", "Left Toe", "Left Camber", "Right Toe", "Right Camber", "Total Toe", "Date Scanned"])
            df = pd.read_csv(self.axle_database_path, dtype=str)
            if self.axle_id not in df["Axle ID"].values:
                pd.concat([df, pd.DataFrame([{"Axle ID": self.axle_id}])], ignore_index=True).to_csv(self.axle_database_path, index=False)
                self.update_status(f"Axle ID {self.axle_id} added to database.")
            else:
                self.update_status(f"Axle ID {self.axle_id} already exists.")

            self.show_hub_scan_screen()

        def content(frame):
            self.barcode_entry = ctk.CTkEntry(frame, placeholder_text="Enter Axle Identifier", width=self.default_button_size)
            self.barcode_entry.pack(pady=(0, 20))
            ctk.CTkButton(frame, text="Enter", command=save_axleID, width=self.default_button_size).pack(pady=(0, 20))
            self.master.bind("<Return>", lambda event: save_axleID())

        self.setup_screen("TorFlex Axle — Measure Hub Alignment", content)

    def measure_arm(self):
        self.type = 'arm'
        """Show screen to enter arm assembly ID for arm measurement."""
        def save_armID():
            self.arm_id = self.barcode_entry.get().strip()
            if not self.arm_id:
                messagebox.showerror("Error", "Please enter an Arm ID")
                return

            os.makedirs(os.path.dirname(self.arm_database_path), exist_ok=True)
            self.initialize_csv(self.arm_database_path, ["Arm ID", "Bar X Angle", "Bar Y Angle", "Bar Z Angle", "Spindle X Angle", "Spindle Y Angle", "Spindle Z Angle", "Total Relative Angle", "Date Scanned"])
            df = pd.read_csv(self.arm_database_path, dtype=str)
            if self.arm_id not in df["Arm ID"].values:
                pd.concat([df, pd.DataFrame([{"Arm ID": self.arm_id}])], ignore_index=True).to_csv(self.arm_database_path, index=False)
                self.update_status(f"Arm ID {self.arm_id} added to database.")
            else:
                self.update_status(f"Arm ID {self.arm_id} already exists.")

            self.show_arm_scan_screen()

        def content(frame):
            self.barcode_entry = ctk.CTkEntry(frame, placeholder_text="Enter Arm Identifier", width=self.default_button_size)
            self.barcode_entry.pack(pady=(0, 20))
            ctk.CTkButton(frame, text="Enter", command=save_armID, width=self.default_button_size).pack(pady=(0, 20))
            self.master.bind("<Return>", lambda event: save_armID())

        self.setup_screen("TorFlex Axle — Measure Crank Arm Alignment", content)

    def calibrate_hub(self):
        def content(frame):
            ctk.CTkLabel(frame, text='Current offsets:', font=ctk.CTkFont(size=20, weight='bold')).pack(pady=(20, 40))
            current_offsets = f"Left X offset: {self.calibrationL['x']}°\nLeft Y offset: {self.calibrationL['y']}°\nRight X offset: {self.calibrationR['x']}°\nRight Y offset: {self.calibrationR['y']}°"
            ctk.CTkLabel(frame, text=current_offsets, font=ctk.CTkFont(size=18)).pack(pady=(10, 20))
            ctk.CTkButton(frame, text='Perform new calibration', command=self.input_calibration_axle_data, width=200).pack(pady=(40, 0))
            ctk.CTkButton(frame, text='Reset calibration to zero', command=self.reset_calibration, width=200).pack(pady=(40, 0))
        self.get_hub_calibration()
        self.setup_screen("Hub Alignment Calibration", content)

    def calibrate_arm(self):
        def content(frame):
            ctk.CTkLabel(frame, text="Crank Arm Calibration - Coming Soon", font=ctk.CTkFont(size=18)).pack(pady=(0, 20))
        self.setup_screen("TorFlex Axle — Calibrate Crank Arm", content)


    # Measure hub
    def get_hub_calibration(self):
        os.makedirs(os.path.dirname(self.calibration_path), exist_ok=True)
        self.initialize_csv(self.calibration_path, ["Left Rotation About X", "Left Rotation About Y", "Right Rotation About X", "Right Rotation About Y", "Date"])
        df = pd.read_csv(self.calibration_path)
        if df.empty:
            self.calibrationL = self.calibrationR = {"x": 0, "y": 0}
            self.calibration_date = date.today()
            pd.DataFrame([{"Left Rotation About X": 0, "Left Rotation About Y": 0, "Right Rotation About X": 0, "Right Rotation About Y": 0, "Date": self.calibration_date}]).to_csv(self.calibration_path, index=False)
        else:
            last = df.iloc[-1]
            self.calibration_date = last["Date"]
            self.calibrationL = {"x": float(last["Left Rotation About X"]), "y": float(last["Left Rotation About Y"])}
            self.calibrationR = {"x": float(last["Right Rotation About X"]), "y": float(last["Right Rotation About Y"])}

    def show_hub_scan_screen(self):
        def content(frame):
            ctk.CTkLabel(frame, text=f"Axle ID: {self.axle_id}", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
            ctk.CTkLabel(frame, text=f"Last calibrated: {self.calibration_date}", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
            ctk.CTkButton(frame, text="Start Scanner", command=self.run_scanner, width=200).pack(pady=(40, 0))
            scan_frame = ctk.CTkFrame(frame)
            scan_frame.pack(pady=(40, 0))
            ctk.CTkButton(scan_frame, text="Measure from existing scan:", command=self.validate_file_and_start, width=200).pack(side=ctk.LEFT, padx=(0, 10))
            self.existing_scan_entry = ctk.CTkEntry(scan_frame, placeholder_text="enter scan file path", width=300)
            self.existing_scan_entry.pack(side=ctk.LEFT)
            mode_frame = ctk.CTkFrame(frame)
            mode_frame.pack(pady=(20, 0))
            ctk.CTkLabel(mode_frame, text="Manual Mode:", font=ctk.CTkFont(size=18)).pack(side=ctk.LEFT, padx=(0, 10))
            self.auto_mode_switch = ctk.CTkSwitch(mode_frame, text="Auto/Manual", command=self.update_auto_mode)
            self.auto_mode_switch.pack(side=ctk.LEFT)
            self.auto_flag = self.auto_mode_switch.get() == 0
            ctk.CTkButton(frame, text="Back", command=self.measure_hub, width=200).pack(pady=(40, 0))
            self.master.bind("<Return>", lambda event: self.run_scanner())
        self.setup_screen("TorFlex Axle — Measure Hub Alignment", content)

    def update_auto_mode(self):
        self.auto_flag = self.auto_mode_switch.get() == 0

    def calc_hub_alignment(self):
        def content(frame):
            ctk.CTkLabel(frame, text='Calculating hub alignment...', font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
        self.setup_screen('Processing Data', content, home_button=False)
        self.master.update()

        def compute_alignment():
            try:
                self.get_hub_calibration()
                scan_resultsL = scan_resultsR = MH.main(self.calibrationL, self.hub_scan_fileA, self.auto_flag, self.scan_type, ui=self)
                # scan_resultsR = MH.main(self.calibrationR, self.hub_scan_fileA, self.auto_flag, self.scan_type, ui=self)
                if isinstance(scan_resultsR, dict) and isinstance(scan_resultsL, dict):
                    self.toe_angleL = scan_resultsL.get("toe_angle", "N/A")
                    self.camber_angleL = scan_resultsL.get("camber_angle", "N/A")
                    self.toe_angleR = scan_resultsR.get("toe_angle", "N/A")
                    self.camber_angleR = scan_resultsR.get("camber_angle", "N/A")
                    self.total_toe = self.toe_angleR - self.toe_angleL if isinstance(self.toe_angleL, (int, float)) and isinstance(self.toe_angleR, (int, float)) else "N/A"
                    self.master.after(0, self.show_hub_results)
                else:
                    self.master.after(0, lambda: messagebox.showerror("Error", "Invalid scan results"))
            except Exception as e:
                self.master.after(0, lambda e=e: messagebox.showerror("Error", f"Scan failed: {e}"))

        # Keep UI responsive by scheduling periodic updates
        def update_ui():
            self.master.update()
            if threading.active_count() > 1:  # Check if background thread is still running
                self.master.after(10, update_ui)  # Schedule next update in 10ms

        # Start computation in a background thread
        threading.Thread(target=compute_alignment, daemon=True).start()
        self.master.after(100, update_ui)

    def show_hub_results(self):
        def content(frame):
            try:
                self.save_hub_results()
                self.print_hub_results()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save or print results: {e}")
            ctk.CTkLabel(frame, text="Measured Hub Alignment", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 10))
            ctk.CTkLabel(frame, text=f'Axle ID: {self.axle_id}', font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))
            results = (f'Left Toe:\t\t{self.toe_angleL}°\nLeft Camber:\t{self.camber_angleL}°\n'
                       f'Right Toe:\t{self.toe_angleR}°\nRight Camber:\t{self.camber_angleR}°\n'
                       f'Total Toe:\t{self.total_toe}°')
            ctk.CTkLabel(frame, text=results, font=ctk.CTkFont(size=18), justify="left", anchor="w").pack(pady=(20, 10))
            ctk.CTkButton(frame, text="Measure another axle", command=self.measure_hub).pack(pady=(10, 20))
            ctk.CTkButton(frame, text='Redo calculation in Manual Mode', command=lambda: [setattr(self, 'auto_flag', False), self.calc_hub_alignment()]).pack(pady=(10, 20))
            self.master.bind("<Return>", lambda event: self.measure_hub())
        self.setup_screen("Results", content)

    def save_hub_results(self):
        df = pd.read_csv(self.axle_database_path, dtype=str)
        df.loc[df["Axle ID"] == self.axle_id, ["Left Toe", "Left Camber", "Right Toe", "Right Camber", "Total Toe", "Date Scanned"]] = [
            self.toe_angleL, self.camber_angleL, self.toe_angleR, self.camber_angleR, self.total_toe, date.today()
        ]
        df.to_csv(self.axle_database_path, index=False)
        self.update_status(f"Scan results saved for Axle ID {self.axle_id}")

    def print_hub_results(self):
        pdf_path = os.path.join(r"C:\Users\Public\CapstoneUI", f"{self.axle_id}.pdf")
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        c = canvas.Canvas(pdf_path, pagesize=(2 * 72, 1 * 72))
        c.setFont("Courier", 8)
        text = c.beginText(0.25 * 72, 0.85 * 72)
        text.setLeading(10)
        for line in [f"Axle ID: {self.axle_id}", f"Left Toe: {self.toe_angleL}°", f"Left Camber: {self.camber_angleL}°", 
                     f"Right Toe: {self.toe_angleR}°", f"Right Camber: {self.camber_angleR}°", f"Total Toe: {self.total_toe}°"]:
            text.textLine(line)
        c.drawText(text)
        c.save()
        webbrowser.open(pdf_path)


    # Measure arm
    def show_arm_scan_screen(self):
        def content(frame):
            # Display Arm ID
            ctk.CTkLabel(frame, text=f"Arm ID: {self.arm_id}", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
            # ctk.CTkLabel(frame, text=f"Last calibrated: {self.calibration_date}", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))

            # Start Scanner Button
            ctk.CTkButton(frame, text="Start Scanner", command=self.run_scanner, width=200).pack(pady=(40, 0))
            scan_frame = ctk.CTkFrame(frame)
            scan_frame.pack(pady=(40, 0))

            # Measure from Existing Scan Button and Path Entry
            ctk.CTkButton(scan_frame, text="Measure from existing scan:", command=self.validate_file_and_start, width=200).pack(side=ctk.LEFT, padx=(0, 10))
            self.existing_scan_entry = ctk.CTkEntry(scan_frame, placeholder_text="enter scan file path", width=300)
            self.existing_scan_entry.pack(side=ctk.LEFT)
            mode_frame = ctk.CTkFrame(frame)
            mode_frame.pack(pady=(20, 0))

            # Manual Mode Toggle
            ctk.CTkLabel(mode_frame, text="Manual Mode:", font=ctk.CTkFont(size=18)).pack(side=ctk.LEFT, padx=(0, 10))
            self.auto_mode_switch = ctk.CTkSwitch(mode_frame, text="Auto/Manual", command=self.update_auto_mode)
            self.auto_mode_switch.pack(side=ctk.LEFT)
            self.auto_flag = self.auto_mode_switch.get() == 0

            # Back Button
            ctk.CTkButton(frame, text="Back", command=self.measure_arm, width=200).pack(pady=(40, 0))
            self.master.bind("<Return>", lambda event: self.run_scanner())
        
        self.setup_screen("TorFlex Axle — Measure Crank Arm Alignment", content)

    def calc_arm_alignment(self):
        def content(frame):
            ctk.CTkLabel(frame, text='Calculating crank arm alignment...', font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
        self.setup_screen('Processing Data', content, home_button=False)
        self.master.update()

        def compute_alignment():
            try:
                # self.get_arm_calibration()
                scan_results = MA.main(self.arm_scan_fileA, self.auto_flag, self.scan_type, ui=self)
                # scan_resultsR = MH.main(self.calibrationR, self.hub_scan_fileA, self.auto_flag, self.scan_type, ui=self)
                if isinstance(scan_results, dict) and isinstance(scan_results, dict):
                    self.total_arm_angle = scan_results.get("total_angle", "N/A")

                    # Bar Angles
                    self.bar_x_angle = scan_results.get("bar_x_angle", "N/A")
                    self.bar_y_angle = scan_results.get("bar_y_angle", "N/A")
                    self.bar_z_angle = scan_results.get("bar_z_angle", "N/A")

                    # Spindle Angles
                    self.spindle_x_angle = scan_results.get("spindle_x_angle", "N/A")
                    self.spindle_y_angle = scan_results.get("spindle_y_angle", "N/A")
                    self.spindle_z_angle = scan_results.get("spindle_z_angle", "N/A")

                    self.master.after(0, self.show_arm_results)
                else:
                    self.master.after(0, lambda: messagebox.showerror("Error", "Invalid scan results"))
            except Exception as e:
                self.master.after(0, lambda e=e: messagebox.showerror("Error", f"Scan failed: {e}"))

        # Keep UI responsive by scheduling periodic updates
        def update_ui():
            self.master.update()
            if threading.active_count() > 1:  # Check if background thread is still running
                self.master.after(10, update_ui)  # Schedule next update in 10ms

        # Start computation in a background thread
        threading.Thread(target=compute_alignment, daemon=True).start()
        self.master.after(100, update_ui)

    def show_arm_results(self):
        def content(frame):
            try:
                self.save_arm_results()
                self.print_arm_results()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save or print results: {e}")
            ctk.CTkLabel(frame, text="Measured Arm Alignment", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 10))
            ctk.CTkLabel(frame, text=f'Arm ID: {self.arm_id}', font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))
            results = (f'Total Relative Angle:\t{self.total_arm_angle}°')
            ctk.CTkLabel(frame, text=results, font=ctk.CTkFont(size=18), justify="left", anchor="w").pack(pady=(20, 10))
            ctk.CTkButton(frame, text="Measure another arm", command=self.measure_arm).pack(pady=(10, 20))
            ctk.CTkButton(frame, text='Redo calculation in Manual Mode', command=lambda: [setattr(self, 'auto_flag', False), self.calc_arm_alignment()]).pack(pady=(10, 20))
            self.master.bind("<Return>", lambda event: self.measure_arm())
        self.setup_screen("Results", content)

    def save_arm_results(self):
        df = pd.read_csv(self.arm_database_path, dtype=str)
        df.loc[df["Arm ID"] == self.arm_id, ["Bar X Angle", "Bar Y Angle", "Bar Z Angle", "Spindle X Angle", "Spindle Y Angle", "Spindle Z Angle", "Total Relative Angle", "Date Scanned"]] = [self.bar_x_angle, self.bar_y_angle, self.bar_z_angle, self.spindle_x_angle, self.spindle_y_angle, self.spindle_z_angle, self.total_arm_angle, date.today()]
        df.to_csv(self.arm_database_path, index=False)
        self.update_status(f"Scan results saved for Arm ID {self.arm_id}")

    def print_arm_results(self):
        pdf_path = os.path.join(r"C:\Users\Public\CapstoneUI", f"{self.arm_id}.pdf")
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        c = canvas.Canvas(pdf_path, pagesize=(2 * 72, 1 * 72))
        c.setFont("Courier", 8)
        text = c.beginText(0.25 * 72, 0.85 * 72)
        text.setLeading(10)
        for line in [f"Arm ID: {self.arm_id}", f"Total Relative Angle: {self.total_arm_angle}°"]:
            text.textLine(line)
        c.drawText(text)
        c.save()
        webbrowser.open(pdf_path)


    # Calibrate hub
    def input_calibration_axle_data(self):
        def content(frame):
            ctk.CTkLabel(frame, text='Enter known values of calibration axle (degrees):', font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 40))
            self.toeL_entry = ctk.CTkEntry(frame, placeholder_text="Left Toe", width=300)
            self.toeL_entry.pack(pady=(0, 20))
            self.camberL_entry = ctk.CTkEntry(frame, placeholder_text="Left Camber", width=300)
            self.camberL_entry.pack(pady=(0, 20))
            self.toeR_entry = ctk.CTkEntry(frame, placeholder_text="Right Toe", width=300)
            self.toeR_entry.pack(pady=(0, 20))
            self.camberR_entry = ctk.CTkEntry(frame, placeholder_text="Right Camber", width=300)
            self.camberR_entry.pack(pady=(0, 20))
            ctk.CTkButton(frame, text="Start Calibration Scan", command=self.start_calibration_scan, width=200).pack(pady=(40, 0))
            self.master.bind("<Return>", lambda event: self.start_calibration_scan())
            mode_frame = ctk.CTkFrame(frame)
            mode_frame.pack(pady=(20, 0))
            ctk.CTkLabel(mode_frame, text="Manual Mode:", font=ctk.CTkFont(size=18)).pack(side=ctk.LEFT, padx=(0, 10))
            self.auto_mode_switch = ctk.CTkSwitch(mode_frame, text="Auto/Manual", command=self.update_auto_mode)
            self.auto_mode_switch.pack(side=ctk.LEFT)
            self.auto_flag = self.auto_mode_switch.get() == 0
        self.setup_screen("Hub Alignment Calibration", content)

    def start_calibration_scan(self):
        try:
            self.offset_toeL, self.offset_camberL = float(self.toeL_entry.get()), float(self.camberL_entry.get())
            self.offset_toeR, self.offset_camberR = float(self.toeR_entry.get()), float(self.camberR_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for toe and camber.")
            return

        def content(frame):
            ctk.CTkLabel(frame, text="Scanning calibration axle...", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
        self.setup_screen("Hub Alignment Calibration", content, home_button=False)
        self.master.update()

        data = PS.perform_scan().astype(float)
        for i in data:
            i = (i - 2**15) * .0102
        np.savetxt(self.temp_scan_pathA, data, delimiter=',', header='X Y Z')
        self.hub_scan_fileA = self.temp_scan_pathA
        self.scan_type = 'live'
        self.calc_calibration()

    def calc_calibration(self):
        def content(frame):
            ctk.CTkLabel(frame, text="Processing calibration...", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 40))
        self.setup_screen("Hub Alignment Calibration", content, home_button=False)

        def compute_calibration():
            try:
                scan_resultsL = scan_resultsR = MH.main(self.calibrationL, self.hub_scan_fileA, self.auto_flag, self.scan_type, ui=self)
                # scan_resultsR = MH.main(self.calibrationR, self.hub_scan_fileA, self.auto_flag, self.scan_type, ui=self)
                if isinstance(scan_resultsR, dict) and isinstance(scan_resultsL, dict):
                    self.rotationL_aboutx = scan_resultsL.get("camber_angle", 0) - self.offset_camberL
                    self.rotationL_abouty = scan_resultsL.get("toe_angle", 0) - self.offset_toeL
                    self.rotationR_aboutx = scan_resultsR.get("camber_angle", 0) - self.offset_camberR
                    self.rotationR_abouty = scan_resultsR.get("toe_angle", 0) - self.offset_toeR
                    self.master.after(0, self.show_calibration_results)
                else:
                    self.master.after(0, lambda: messagebox.showerror("Error", "Invalid calibration results"))
            except Exception as e:
                self.master.after(0, lambda e=e: messagebox.showerror("Error", f"Calibration failed: {e}"))

        # Keep UI responsive by scheduling periodic updates
        def update_ui():
            self.master.update()
            if threading.active_count() > 1:  # Check if background thread is still running
                self.master.after(10, update_ui)  # Schedule next update in 10ms

        # Start computation in a background thread
        threading.Thread(target=compute_calibration, daemon=True).start()
        self.master.after(100, update_ui)

    def show_calibration_results(self):
        def content(frame):
            try:
                self.save_calibration()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save calibration: {e}")
            ctk.CTkLabel(frame, text="Results — offsets will be applied to all future measurements:", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 10))
            results = f"Left X offset: {self.rotationL_aboutx}°\nLeft Y offset: {self.rotationL_abouty}°\nRight X offset: {self.rotationR_aboutx}°\nRight Y offset: {self.rotationR_abouty}°"
            ctk.CTkLabel(frame, text=results, font=ctk.CTkFont(size=18)).pack(pady=(10, 20))
        self.setup_screen("Hub Alignment Calibration", content)

    def save_calibration(self):
        self.initialize_csv(self.calibration_path, ["Left Rotation About X", "Left Rotation About Y", "Right Rotation About X", "Right Rotation About Y", "Date"])
        pd.concat([pd.read_csv(self.calibration_path, dtype=str), pd.DataFrame([{
            "Left Rotation About X": self.rotationL_aboutx, "Left Rotation About Y": self.rotationL_abouty,
            "Right Rotation About X": self.rotationR_aboutx, "Right Rotation About Y": self.rotationR_abouty,
            "Date": date.today()
        }])], ignore_index=True).to_csv(self.calibration_path, index=False)
        self.update_status("Calibration saved")

    def reset_calibration(self):
        try:
            os.remove(self.calibration_path)
            self.get_hub_calibration()
            self.calibrate_hub()
        except FileNotFoundError:
            self.update_status(f"Calibration file {self.calibration_path} not found, resetting to default")
            self.get_hub_calibration()
            self.calibrate_hub()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset calibration: {e}")

    def initialize_csv(self, path, columns):
        if not os.path.exists(path):
            pd.DataFrame(columns=columns).to_csv(path, index=False)

if __name__ == "__main__":
    root = ctk.CTk()
    app = Dexter_Capstone_UI(root)
    root.mainloop()
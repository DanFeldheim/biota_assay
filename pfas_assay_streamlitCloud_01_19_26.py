#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 09:36:23 2026

An app to analyze data collected with Biota's ttr pfas assay.

@author: danfeldheim
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

# # Clean up any leftover figures or memory at the start of each run
# plt.close("all")
# gc.collect()

# # Checks memory usage
# # Called from navigation function, which is commented out when not checking for memory leaks
# def get_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss / (1024 ** 2)


class Flow_Control():
    """This class makes all of the calls to other classes and methods."""
    
    def __init__(self):
        
        # Store fit parameters and standard concentrations in a dict
        self.params_dict = {
                            # 4PL fit parameters from calibration curve
                            'A':20272.331424748776,
                            'B':52698.72476352284,
                            'C':449.1635703559516,
                            'D':1.0622090103190498,
                            # Standards concentrations
                            'std1_set_conc':10,
                            'std2_set_conc':50
                            }
         
        
    def all_calls(self):
        """This is the main logic workflow. All calls to other functions are here."""
        
        #---------------------------------------------------------------------------------------------
        # Set up the header, login form, and check user credentials
        
        # Create Setup instance
        setup = Setup()
        
        # Render the header
        header = setup.header()
        
        # This is only to show the memory usage to check for memory leak
        # nav_bar = setup.navigation()
       
        # Create Load_Data instance
        data_loader = Load_Data()
        
        # Upload button
        file_upload = data_loader.upload()
        
        if file_upload is not None:
            # Create dict with filename as key and data as value
            dfs_dict = data_loader.create_df(file_upload)
            # st.write(dfs_dict)
            
        # Create class instance
        conc_calcs = Calculate_Concs()
        
        # Calculate concentrations from RFU values and fit params
        concs_dict = conc_calcs.invert_4pl(dfs_dict, self.params_dict)
        
        # Check the standards and adjust their RFU values if threshold is exceeded.
        stds_checked_adjusted_data_dict = conc_calcs.analyze_standards(concs_dict, self.params_dict)
        adjusted_data_dict = conc_calcs.invert_4pl(
                                                    stds_checked_adjusted_data_dict,
                                                    self.params_dict,
                                                    rfu_col="adjusted_RFU",
                                                    conc_col="adjusted_concs",
                                                    flag_col="adjusted_flag"
                                                  ) 
        # Create class instance
        display = Display_Results()
        # Generate and display a results table
        results_table = display.create_table(adjusted_data_dict)
        
        # Download results
        download = display.download_results(results_table)
        
        
        
class Setup():
    """Class that lays out the app header and sidebar."""
    
    def __init__(self):
        
        pass
    
    def header(self):
        
        # Draw line across the page
        st.divider()
        
        # Add a logo and title
        col1, col2, col3, col4, col5 = st.columns([0.75,1.25,0.25,4,3])
        
        with col2:
        
            st.image(st.session_state['logo'])
            
        with col4:

            # st.write('')
            # st.write('')
            st.write('')
            st.write('')
                
            st.markdown(f"<p style='color: DarkBlue; \
                  font-size: 36px; \
                  margin: 0;'>PFAS Biosensor Data Analysis Tool</p>",
                  unsafe_allow_html=True)
            
        st.divider()    
    
    # def navigation(self):
    #     """Creates a sidebar only for the purposes of checking memory leak. Otherwise not called."""
        
    #     with st.sidebar:
            
    #         st.write('')
    #         st.write('')

    #         st.subheader("Diagnostics")
    #         st.metric("Memory (MB)", f"{get_memory_usage():.2f}")
            
    #         col1, col2, col3 = st.columns([1,8,1])
    #         with col2:
                
    #             st.markdown(f"<p style='color: DarkBlue; \
    #                   font-size: 28px; \
    #                   margin: 0;'>Options Menu</p>",
    #                   unsafe_allow_html=True)
                
    #         st.divider()    

class Load_Data():
    """Creates a file uploader and imports data as dataframe."""
    
    def __init__(self):
        
        pass
    
    def upload(self):
        
        col1, col2 = st.columns([1,1])
        
        with col1:
        
            # File uploader that allows multiple files
            uploaded_files = st.file_uploader(
                                              "Upload Denovix files", 
                                              type=["csv"],  
                                              accept_multiple_files=True, 
                                              )
            
        st.divider()
        
        
        return uploaded_files
    
    def create_df(self, files):
        
        data_dict = {}
        
        # Loop through files, create dfs, and add to data_dict with filename as key without .csv
        for file in files:
            
            # Get filename but strip the .csv
            filename = file.name[:-4]
            df = pd.read_csv(file)
            selected_cols = ['Sample Name', 'Blue RFU']
            df = df[selected_cols]
            data_dict[filename] = df
            
        return data_dict
    
    
class Calculate_Concs():
    
    def __init__(self):
        
        pass         
   
    # def invert_4pl(self, data_dict, params_dict, rfu_col="Blue RFU"):
    def invert_4pl(self, 
                   data_dict, 
                   params_dict, 
                   rfu_col="Blue RFU",
                   conc_col="Concentration", 
                   flag_col="Conc Flag"):
        
        # Store results
        results_dict = {}
    
        # Get the low and high A/B params. These set the range of cal curve
        low = min(params_dict['A'], params_dict['B'])
        high = max(params_dict['A'], params_dict['B'])
    
        # Loop through filenames
        for filename, df in data_dict.items():
            
            df = df.copy()
    
            # Get rfu values as floats
            rfu = df[rfu_col].astype(float).to_numpy()
    
            # Start with everything in the np array invalid until proven otherwise
            conc = np.full(rfu.shape, np.nan, dtype=float)
            # Start a flag np array with all elements set to pfas detected
            flag = np.full(rfu.shape, "PFAS Detected", dtype=object)
    
            # If rfu <= the low and high range, the np array gets reset
            below = rfu <= low
            above = rfu >= high
            flag[below] = "Below Assay LLOD"
            flag[above] = "PFAS Detected Above Assay ULOD"
    
            # Only compute for in-range values
            in_range = ~(below | above) & np.isfinite(rfu)
    
            # Compute ratio and validity
            # Create np array with same shape as rfu and filled with nan
            ratio = np.full(rfu.shape, np.nan, dtype=float)
            # Do calculation for rfu in the correct range
            ratio[in_range] = (params_dict['B'] - rfu[in_range]) / (rfu[in_range] - params_dict['A'])
            
            # Define a valid ratio
            valid_ratio = in_range & np.isfinite(ratio) & (ratio > 0) \
                                    & np.isfinite(params_dict['C']) \
                                    & (params_dict['C'] > 0) \
                                    & np.isfinite(params_dict['D']) \
                                    & (params_dict['D'] != 0)
    
            # Compute concentration where valid
            conc[valid_ratio] = params_dict['C'] * (ratio[valid_ratio] ** (1.0 / params_dict['D']))
    
            # Mark remaining in-range-but-invalid as invalid
            invalid = in_range & ~valid_ratio
            flag[invalid] = "Undetermined"
    
            # If conc calculation produced non-finite/negative, mark invalid
            bad_conc = valid_ratio & (~np.isfinite(conc) | (conc < 0))
            flag[bad_conc] = "Undetermined"
            conc[bad_conc] = np.nan
    
            df[conc_col] = conc
            df[flag_col] = flag
    
            results_dict[filename] = df
    
        return results_dict
    
    def analyze_standards(self, 
                          dfs_dict, 
                          params_dict):
        
        """
        Checks the measured standards against the known concentrations and adjusts rfu values
        if they exceed 7.5% difference.
        """
        
        adjusted_data_dict = {}
        
        for filename, df in dfs_dict.items(): 
            
            df = df.copy()
            
            # Create a new column for adjusted RFU values and fill with original RFU values
            # If adjustment is needed, the values will change.
            df["adjusted_RFU"] = pd.to_numeric(df["Blue RFU"], errors="coerce")
            
            # Define columns for error messages to appear
            col1, col2 = st.columns([1.5,1])
            
            # Get the standard concentration values as pandas series
            std1_measured = df.loc[df["Sample Name"] == "std1", "Concentration"]
            std2_measured = df.loc[df["Sample Name"] == "std2", "Concentration"]
            
            if std1_measured.empty or std2_measured.empty:
                
                with col1:
                    
                    st.error('One or both standards is missing.')
                    
                # If the standard is missing, stick the original df in adjusted_data_dict and move on.
                # Make sure the df gets the adjusted_rfu column
                adjusted_data_dict[filename] = df  
                    
                continue
            
            # Get the values from the series
            std1_measured = float(std1_measured.values[0])
            std2_measured = float(std2_measured.values[0])
            
            # Check standards exist
            if std1_measured == 0 or std2_measured == 0:
                
                with col1:
                    
                    st.error('At least one of the standards is 0.')
                
                # Make sure the df gets the adjusted_rfu column
                adjusted_data_dict[filename] = df  
                
                continue  
    
            # Check order
            if std1_measured >= std2_measured:
                
                with col1:
                    st.error(
                            'Oops! Something is wrong with the values of your standards. \
                            Please check that they were prepared and labeled correctly.'
                            )
                        
                # Make sure the df gets the adjusted_rfu column
                adjusted_data_dict[filename] = df  
                
                continue 
            
            # Set a 7.5% error threshold above which the RFUs will be adjusted.
            threshold = 0.075  

            std1_off = abs(std1_measured - params_dict['std1_set_conc']) / params_dict['std1_set_conc']
            std2_off = abs(std2_measured - params_dict['std2_set_conc']) / params_dict['std2_set_conc']
            
            # If the tolerance was exceeded, adjust the RFU values and recalculate concentrations
            if (std1_off > threshold) or (std2_off > threshold):
                
                # Get measured RFU values for the standards (from this file) 
                std1_rfu_s = pd.to_numeric(
                                            df.loc[df["Sample Name"] == "std1", "Blue RFU"], 
                                            errors="coerce").dropna()
                
                std2_rfu_s = pd.to_numeric(
                                            df.loc[df["Sample Name"] == "std2", "Blue RFU"], 
                                            errors="coerce").dropna()

                if std1_rfu_s.empty or std2_rfu_s.empty:
                    
                    with col1:
                        
                        st.error("One of your standard RFU values is missing.")
                    
                    # Make sure the df gets the adjusted_rfu column
                    adjusted_data_dict[filename] = df  
                    
                    continue

                std1_rfu = float(std1_rfu_s.values[0])
                std2_rfu = float(std2_rfu_s.values[0])

                # Calculate what the RFU values should have been for the standards
                # given their known concentrations.
                # Use a linear correction: yobs = alpha + beta * ytrue
                # alpha accounts for all values being shifted up or down by the same amount
                # due to background fluorescence, for instance.
                # beta accounts for detector sensitivity changes.
                # Since there are 2 standards, subtracting them leaves
                # beta = (y2,obs - y1,obs)/(y2,pred -y1,pred)
                # alpha and beta are calculated below.
                A = params_dict["A"]; B = params_dict["B"]; C = params_dict["C"]; D = params_dict["D"]

                y1_pred = self.four_pl(params_dict["std1_set_conc"], A, B, C, D)
                y2_pred = self.four_pl(params_dict["std2_set_conc"], A, B, C, D)

                denom = (y2_pred - y1_pred)
                
                # Catch division by 0 errors
                if denom == 0 or not np.isfinite(denom):
                    
                    with col1:
                        
                        st.error("Cannot apply RFU correction (predicted standard RFUs are identical).")
                    
                    # Make sure the df gets the adjusted_rfu column
                    adjusted_data_dict[filename] = df  
                    
                    continue

                # y_obs ≈ alpha + beta * y_true
                # Calculate beta as described above
                beta = (std2_rfu - std1_rfu) / denom
                # Calculate alpha using beta
                alpha = std1_rfu - beta * y1_pred

                if (not np.isfinite(alpha)) or (not np.isfinite(beta)) or beta <= 0:
                    
                    with col1:
                        
                        st.error(f"Cannot apply RFU correction to {filename}.")
                    
                    # Make sure the df gets the adjusted_rfu column
                    adjusted_data_dict[filename] = df  
                    
                    continue

                # Create adjusted RFU column for ALL rows
                rfu = pd.to_numeric(df["Blue RFU"], errors="coerce")
                df["adjusted_RFU"] = (rfu - alpha) / beta

            adjusted_data_dict[filename] = df
            
        return adjusted_data_dict
            
    def four_pl(self, x, A, B, C, D):
        """
        Forward 4-parameter logistic (4PL):
        x = concentration
        returns predicted RFU (fluorescence)
    
        A = bottom (low asymptote)
        B = top (high asymptote)
        C = inflection (EC50)
        D = Hill slope
        """
        x = np.asarray(x, dtype=float)
        return A + (B - A) / (1.0 + (x / C) ** D)

           
class Display_Results():
    
    def __init__(self):
        
        pass
    
    def create_table(self, results_dict):
        
        # Create a dict for the dataframes as they appear in the table.
        edited_tables = {}
        
        # Write the filename
        st.markdown(f"<p style='color: DarkRed; \
              font-size: 24px; \
              margin: 0;'>Results Tables</p>",
              unsafe_allow_html=True)
            
        st.write('')
        
        for file_name, df in results_dict.items():
            
            st.write('')
            
            # st.write(df)
            # st.stop()
        
            # Write the filename
            st.markdown(f"<p style='color: Black; \
                  font-size: 18px; \
                  margin: 0;'>File: {file_name}</p>",
                  unsafe_allow_html=True)
                
            display_df = df[[
                            "Sample Name",
                            "Blue RFU",
                            "adjusted_RFU",
                            "adjusted_concs",
                            "adjusted_flag"
                           ]].copy()
                        
            display_df = display_df.rename(columns={
                                                    "Blue RFU": "Measured RFU",
                                                    "adjusted_RFU": "Adjusted RFU",
                                                    "adjusted_concs": "Total PFAS Concentration",
                                                    "adjusted_flag": "Sample Status"
                                                   })
            
            # Create column_config for all columns as "small"
            column_config = {
                            col: st.column_config.Column(col, width=225)
                            for col in display_df.columns
                            }
                
            data_df = st.data_editor(display_df, 
                                     use_container_width=False,
                                     num_rows="static",
                                     hide_index=True,
                                     height=350,
                                     column_config=column_config,
                                     key=f"raw{file_name}"
                                     )
            
            st.divider()
            
            edited_tables[file_name] = data_df
            
        return edited_tables
            
            
    def download_results(self, edited_tables, filename="PFAS_results_all_files.csv"):

        if not edited_tables:
            st.warning("No results available to download.")
            return
    
        combined = []
        for file_name, df in edited_tables.items():
            d = df.copy()
            d.insert(0, "File", file_name)
            combined.append(d)
    
        combined_df = pd.concat(combined, ignore_index=True)
        csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
    
        st.download_button(
                            label="Download Results",
                            data=csv_bytes,
                            file_name=filename,
                            mime="text/csv",
                          )
                    
            
            
        
       
                
            
   

# Run 
if __name__ == '__main__':
    
    # Get the path relative to the current file (inside Docker container)
    BASE_DIR = os.path.dirname(__file__)
    
    if 'directory' not in st.session_state:
        st.session_state['directory'] = BASE_DIR
        
    # Use this for cloud
    st.session_state['logo'] = 'mote_logo.png'
        
    # Load image for favicon
    logo_img = Image.open(st.session_state['logo'])
        
    # Page config
    st.set_page_config(layout = "wide", 
                       page_title = 'Biota', 
                       page_icon = logo_img,
                       initial_sidebar_state="auto", 
                       menu_items = None)
    
    
    # Call Flow_Control class that makes all calls to other classes and methods
    obj1 = Flow_Control()
    all_calls = obj1.all_calls()

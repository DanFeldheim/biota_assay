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
import base64
from textwrap import dedent

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
                            'A':3096.082204433693,
                            'B':6423.412709072206,
                            'C':286.8973173597928,
                            'D':1.61008049679831,
                            # Standards concentrations in nM
                            'std1_set_conc':10,
                            'std2_set_conc':5000,
                            # PFOA MW for conversion to ppb
                            'pfoa_MW':413.06,
                            # Set an LLOD of 60 nM in RFU units
                            # Use the 4pl fit params to convert 60 nM to RFU
                            # Any RFU larger than this will not be detectable
                            'RFU_LOD':6526
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
        
        ppb_dict = conc_calcs.convert_to_ppb(adjusted_data_dict, self.params_dict['pfoa_MW'])
        
        # Perform a replicate analysis
        replicates_dict = conc_calcs.replicate_analysis(ppb_dict)
        
        # Combine ppb_dict and replicates_dict
        all_results_dict = {
                            filename: {
                                       "ppb_table": ppb_dict[filename],
                                       "replicates_table": replicates_dict[filename],
                                      }
                            for filename in ppb_dict
                           }
        
        # Create class instance
        display = Display_Results()
        
        # Generate and display a summary results and replicate analysis tables
        ppb_results_table, replicates_results_table = display.create_tables(all_results_dict)
        
        # Download results
        download = display.download_results(ppb_results_table, replicates_results_table)
        
        
        
class Setup():
    """Class that lays out the app header and sidebar."""
    
    def __init__(self):
        
        pass
    
    
    def header(self):

        logo_path = st.session_state["logo"]
    
        with open(logo_path, "rb") as f:
            
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")
    
        # Tune these to match your logo + spacing
        logo_height_px = 90
        gap_px = 20
        left_padding_px = 24
    
        # Approximate logo width based on its rendered height.
        approx_logo_width_px = 110
    
        text_left_px = left_padding_px + approx_logo_width_px + gap_px
    
        st.markdown(dedent(f"""
                            <!-- Blue banner with logo only -->
                            <div style="
                              width: 100vw;
                              margin-left: calc(-50vw + 50%);
                              background-color: #0033A0;
                              padding: 14px 0px;
                            ">
                              <div style="
                                max-width: 1100px;
                                margin: 0 auto;
                                padding: 0 {left_padding_px}px;
                                display: flex;
                                align-items: center;
                              ">
                                <img src="data:image/png;base64,{logo_base64}"
                                     style="height: {logo_height_px}px; width: auto;" />
                              </div>
                            </div>
                        
                            <!-- White area text, shifted right to sit under/right of logo -->
                            <div style="max-width: 1100px; margin: 12px auto 0 auto; padding: 0 {left_padding_px}px;">
                              <div style="padding-left: {text_left_px - left_padding_px}px;">
                                <p style="
                                  color: DarkBlue;
                                  font-size: 36px;
                                  margin: 0;
                                  font-family: 'DIN Alternate', Arial, sans-serif;
                                  line-height: 1.1;
                                ">
                                  RapidTest<sup style="font-size:60%;">TM</sup><br>
                                  PFAS<br>
                                  Data Analysis Tool
                                </p>
                              </div>
                            </div>
                            """), unsafe_allow_html=True)
    
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

            # Extract relevant columns
            selected_cols = ['Sample Name', 'Blue RFU']
            df = df[selected_cols]

            # Strip out "," and convert RFU to float
            # df["Blue RFU"] = (
                              # df["Blue RFU"]
                              # .astype(str)
                              # .str.replace(",", "", regex=False)
                              # .str.strip()
                             # )

            blue_string = df["Blue RFU"].astype("string")

            df["Blue RFU"] = (
                              blue_string
                              .str.replace("\u00a0", "", regex=False)   # NBSP
                              .str.replace("\u200b", "", regex=False)   # zero-width space
                              .str.replace(",", "", regex=False)
                              .str.strip()
                             )
            
            df["Blue RFU"] = pd.to_numeric(df["Blue RFU"], errors="coerce").fillna(0).astype(float)
            
            data_dict[filename] = df
            
        return data_dict
    
    
class Calculate_Concs():
    
    def __init__(self):
        
        pass         
   
    def invert_4pl(self, 
                   data_dict, 
                   params_dict, 
                   rfu_col="Blue RFU",
                   conc_col="Concentration", 
                   flag_col="Conc Flag"):
        
        # Store results
        results_dict = {}
    
        # Get the low and high A/B params. These set the range of cal curve (LLOD/ULOD)
        low = min(params_dict['A'], params_dict['B'])
        # high = max(params_dict['A'], params_dict['B'])
        # Use the LOD_RFU in params_dict to set the llod
        high = params_dict['RFU_LOD']
    
        # Loop through filenames
        for filename, df in data_dict.items():
            
            df = df.copy()
    
            # Get rfu values as floats
            rfu = df[rfu_col].astype(float).to_numpy()
    
            # Start with everything in the np array invalid until proven otherwise
            conc = np.full(rfu.shape, np.nan, dtype=float)
            # Start a flag np array with all elements set to pfas detected
            flag = np.full(rfu.shape, "PFAS Detected in Quantifiable Range", dtype=object)
    
            # If rfu <= the low and high range, the np array gets reset
            below = rfu <= low
            above = rfu >= high
            flag[below] = "PFAS Detected Above Quantifiable Range"
            flag[above] = "Below Assay LLOD"
    
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
            # Catch all forms of sample name like Std1, STD1, STD 1
            df["Sample Name"] = df["Sample Name"].astype(str).str.strip().str.lower()
            # Get conc
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
    
    def convert_to_ppb(self, results_dict, mw):
        
        ppb_dict = {}
        
        for filename, df in results_dict.items():
            
            df['conc_ppb'] = df['adjusted_concs'] * 0.413
            
            ppb_dict[filename] = df
            
        return ppb_dict
    
    def replicate_analysis(self, results_dict):
        
        replicate_results_dict = {}
        
        for filename, df in results_dict.items():
            
            # Create a column containing all characters to the left of "_".
            df["sample_id"] = df["Sample Name"].str.partition("_")[0]
            
            # Remove standards
            df = df[~df["Sample Name"].str.contains("std", case=False, na=False)]
            
            replicate_df = df[['Sample Name', 'sample_id', 'conc_ppb']].copy()
            
            # Ensure conc_ppb is numeric; None/"", etc -> NaN
            replicate_df["conc_ppb"] = pd.to_numeric(replicate_df["conc_ppb"], errors="coerce")
            
            summary_df = (
                        df.groupby(["sample_id"], as_index=False)
                          .agg(
                              sample_name=("Sample Name", "first"),  
                              n=("conc_ppb", "count"),                
                              mean_ppb=("conc_ppb", "mean"),
                              sd_ppb=("conc_ppb", "std"),           
                              )
                      )
            
            summary_df["rsd_pct"] = 100 * summary_df["sd_ppb"] / summary_df["mean_ppb"]
            
            # avoid inf when mean is 0
            summary_df.loc[summary_df["mean_ppb"] == 0, "rsd_pct"] = np.nan
            
            replicate_results_dict[filename] = summary_df
            
            
        return replicate_results_dict
        

           
class Display_Results():
    
    def __init__(self):
        
        pass
    
    def create_tables(self, all_results_dict):
        
        # Create a dict for the dataframes as they appear in the table.
        edited_ppb_tables = {}
        edited_replicates_tables = {}
        
        # Write header
        st.markdown(f"<p style='color: DarkBlue; \
              font-size: 24px; \
              margin: 0;'>Results Tables</p>",
              unsafe_allow_html=True)
            
        st.write('')
        
        # Render concentration table first
        for filename, df in all_results_dict.items():
            
            st.write('')
          
            # Write the filename
            st.markdown(f"<p style='color: Green; \
                  font-size: 20px; \
                  margin: 0;'>File: {filename}</p>",
                  unsafe_allow_html=True)
                
            st.write('')
                
            # Write the table name
            st.markdown(f"<p style='color: DodgerBlue; \
                  font-size: 18px; \
                  margin: 0;'>Summary Table</p>",
                  unsafe_allow_html=True)
                
            ppb_df = df['ppb_table'][[
                                     "Sample Name",
                                     "Blue RFU",
                                     "adjusted_concs",
                                     "conc_ppb",
                                     "adjusted_flag"
                                    ]].copy()
                        
            ppb_df = ppb_df.rename(columns={
                                             "Blue RFU": "RFU",
                                             "adjusted_concs": "[PFAS](nM)",
                                             "conc_ppb":"[PFAS](ppb)",
                                             "adjusted_flag": "Results Summary"
                                            })
            
            
            # Round numeric values 
            pfas_cols = ["[PFAS](nM)", "[PFAS](ppb)"]
            ppb_df[pfas_cols] = ppb_df[pfas_cols].round(2)
            
            # Replace None/NaN in the PFAS columns with a friendly label
            for col in ["[PFAS](nM)", "[PFAS](ppb)"]:
                ppb_df[col] = ppb_df[col].fillna("Not Quantifiable")
            
            cols = list(ppb_df.columns)

            column_config = {}
            
            for i, col in enumerate(cols):
                
                if i == 0:
                    column_config[col] = st.column_config.Column(col, width=200)
                    
                elif i == len(cols) - 1:
                    column_config[col] = st.column_config.Column(col, width=250)
                    
                elif i == 1:
                    column_config[col] = st.column_config.Column(col, width=100)
                    
                else:
                    column_config[col] = st.column_config.Column(col, width=150)
            
            # Show 10 rows
            visible_rows = min(len(ppb_df), 10)
            height = 38 + visible_rows * 35
                        
            data_df = st.data_editor(ppb_df, 
                                     use_container_width=False,
                                     num_rows="static",
                                     hide_index=True,
                                     height=height,
                                     column_config=column_config,
                                     key=f"raw{filename}"
                                     )
            
            # Store concentration table
            edited_ppb_tables[filename] = data_df
                        
            st.write('')
            st.write('')
            
            # Render replicate table 
            # Write the table name
            st.markdown(f"<p style='color: DodgerBlue; \
                  font-size: 18px; \
                  margin: 0;'>Replicates Analysis Table</p>",
                  unsafe_allow_html=True)
             
            replicates_df = df['replicates_table'][[
                                                     "sample_id",
                                                     "n",
                                                     "mean_ppb",
                                                     "sd_ppb",
                                                     "rsd_pct"
                                                  ]].copy()
            
            # st.write(replicates_df)
            # st.stop()
                
            replicates_df = replicates_df.rename(columns={
                                                            "sample_id": "Sample Name",
                                                            "n":"Replicates",
                                                            "mean_ppb":"Mean [PFAS](ppb)",
                                                            "sd_ppb":"Standard Deviation (ppb)",
                                                            "rsd_pct":"% RSD"
                                                            })
            
            # st.write(replicates_df)
            
            # Round numeric values 
            replicate_cols = ["Standard Deviation (ppb)", "Mean [PFAS](ppb)", "% RSD"]
            replicates_df[replicate_cols] = replicates_df[replicate_cols].round(2)
            
            # Replace None/NaN in the PFAS columns with a friendly label
            for col in ["Standard Deviation (ppb)", "Mean [PFAS](ppb)", "% RSD"]:
                replicates_df[col] = replicates_df[col].fillna("Not Enough Replicates")
                
            cols = list(replicates_df.columns)

            column_config = {}
            
            for i, col in enumerate(cols):
                
                # if i < 3:
                #     column_config[col] = st.column_config.Column(col, width=150)
              
                # else:
                #     column_config[col] = st.column_config.Column(col, width=225)
                
                if i == 0:
                    column_config[col] = st.column_config.Column(col, width=175)
                    
                elif i == len(cols) - 1:
                    column_config[col] = st.column_config.Column(col, width=175)
                    
                elif i == 1:
                    column_config[col] = st.column_config.Column(col, width=100)
                    
                else:
                    column_config[col] = st.column_config.Column(col, width=200)
             
            # Fix the table height according to the number of rows
            visible_rows = min(len(replicates_df), 10)
            height = 38 + visible_rows * 35
                        
            replicates_data_df = st.data_editor(replicates_df, 
                                                use_container_width=False,
                                                num_rows="static",
                                                hide_index=True,
                                                height=height,
                                                column_config=column_config,
                                                key=f"raw{filename}_2"
                                                )
            
            # Store concentration table
            edited_replicates_tables[filename] = replicates_data_df
            
            
        st.divider()    
            
        return edited_ppb_tables, edited_replicates_tables                
            
    def download_results(self, 
                         edited_ppb_tables, 
                         edited_replicates_tables):

        # Check for data
        if not edited_ppb_tables or not edited_replicates_tables:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.info("Upload Denovix data files to begin.")
                
            return
    
        # Generate PPB table
        # Combine results from all files
        combined_ppb_results = []
        
        for filename, df in edited_ppb_tables.items():
            d = df.copy()
            d.insert(0, "File", filename)
            combined_ppb_results.append(d)
    
        combined_ppb_df = pd.concat(combined_ppb_results, ignore_index=True)
        # Convert df to csv
        ppb_csv_str = combined_ppb_df.to_csv(index=False)
    
        #-------------------------------------------------------------------
        # Replicates table 
        # Combine data from all files
        combined_replicates_results = []
        for filename, df in edited_replicates_tables.items():
            r = df.copy()
            r.insert(0, "File", filename)
            combined_replicates_results.append(r)
    
        combined_replicates_df = pd.concat(combined_replicates_results, ignore_index=True)
        # Convert to csv
        replicates_csv_str = combined_replicates_df.to_csv(index=False)
    
        # Download buttons
        st.download_button(
                          label="Download Summary Tables",
                          data=ppb_csv_str,
                          file_name="summary_tables.csv",
                          mime="text/csv",
                          )
    
        st.download_button(
                          label="Download Replicates Tables",
                          data=replicates_csv_str,
                          file_name="replicates_tables.csv",
                          mime="text/csv",
                          )
        
        
  
# Run 
if __name__ == '__main__':
    
    # Get the path relative to the current file (inside Docker container)
    BASE_DIR = os.path.dirname(__file__)
    
    if 'directory' not in st.session_state:
        st.session_state['directory'] = BASE_DIR

    # Use these for cloud
    st.session_state["logo"] = "logo-no-background.png"
    # Store favicon image
    st.session_state["favicon"] = "logo-black.png"
    
    # Use these for local machine
    # if 'logo' not in st.session_state:
        # st.session_state['logo'] = BASE_DIR + '/logo-no-background.png'
        
    # if 'favicon' not in st.session_state:
        # st.session_state['favicon'] = BASE_DIR + '/logo-black.png'
    
    # Load image for favicon
    logo_img = Image.open(st.session_state['favicon'])
        
    # Page config
    st.set_page_config(layout = "wide", 
                       page_title = 'Biota', 
                       page_icon = logo_img,
                       initial_sidebar_state="auto", 
                       menu_items = None)
    
    
    # Call Flow_Control class that makes all calls to other classes and methods
    obj1 = Flow_Control()
    all_calls = obj1.all_calls()

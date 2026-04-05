import h5py
import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import os

# --- UPDATED PATHS FOR ISFAHAN DATASET ---
insar_path = r'C:\Users\DFMRendering\Desktop\subsidence\Revise\Data\Isfahan\000089_028A_05817_131313_filt.hdf5'
era5_path = r'C:\Users\DFMRendering\Desktop\subsidence\Revise\Data\Isfahan\ERA5_Isfahan_MultiTemporal.tif'
soil_path = r'C:\Users\DFMRendering\Desktop\subsidence\Revise\Data\Isfahan\SoilGrids_Isfahan_0_100cm_ERA5Grid_Filtered.tif'
txt_metadata = r'C:\Users\DFMRendering\Desktop\subsidence\Revise\Data\Isfahan\ERA5_Metadata_Summary.txt'
output_npz = r'C:\Users\DFMRendering\Desktop\subsidence\Revise\Data\Isfahan\Merged_Dataset_3D.npz'
output_report = r'C:\Users\DFMRendering\Desktop\subsidence\Revise\Data\Isfahan\Merged_Dataset_Report.txt'


def parse_era5_metadata(txt_path):
    """Parses the text file to identify variables and dates per band."""
    band_map = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "|" in line and "Variable Name" not in line and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6:
                    band_map.append({
                        'idx': int(parts[0]),
                        'var': parts[1],
                        'year': parts[2],
                        'month': parts[3]
                    })
    return pd.DataFrame(band_map)


def align_and_save():
    report = []
    report.append("=== MERGED DATASET ALIGNMENT & VALIDATION REPORT (ISFAHAN) ===\n")

    # 1. Load ERA5 Metadata and TIFF
    print("Parsing ERA5 Metadata...")
    df_era5 = parse_era5_metadata(txt_metadata)
    unique_vars = list(df_era5['var'].unique())
    insar_aux_features = ['coh_avg', 'hgt', 'U.geo', 'mask', 'vstd']

    # Soil feature names expected from exported GeoTIFF band order
    # If band descriptions exist in TIFF, they will overwrite these defaults.
    default_soil_features = [
        'bdod_gcm3',
        'clay_pct',
        'phh2o_pH',
        'sand_pct',
        'silt_pct',
        'soc_dgkg'
    ]

    with rasterio.open(era5_path) as src_era5:
        era5_data = src_era5.read()
        aff = src_era5.transform
        e_rows, e_cols = src_era5.shape

        rows_idx = np.arange(e_rows)
        cols_idx = np.arange(e_cols)
        cols_grid, rows_grid = np.meshgrid(cols_idx, rows_idx)
        lons_full, lats_full = aff * (cols_grid, rows_grid)
        lons_full, lats_full = np.array(lons_full), np.array(lats_full)

    # 1.5 Load SoilGrids TIFF
    print("Loading SoilGrids TIFF...")
    with rasterio.open(soil_path) as src_soil:
        soil_data = src_soil.read()  # shape: (bands, rows, cols)
        s_rows, s_cols = src_soil.shape

        soil_band_names = list(src_soil.descriptions) if src_soil.descriptions is not None else None
        if soil_band_names is None or all([(x is None or str(x).strip() == '') for x in soil_band_names]):
            soil_band_names = default_soil_features
        else:
            soil_band_names = [
                str(name).strip() if (name is not None and str(name).strip() != '') else default_soil_features[i]
                for i, name in enumerate(soil_band_names)
            ]

    if (s_rows != e_rows) or (s_cols != e_cols):
        raise ValueError(
            f"Soil TIFF grid does not match ERA5 grid: "
            f"Soil=({s_rows}, {s_cols}) vs ERA5=({e_rows}, {e_cols})"
        )

    if soil_data.shape[0] != len(soil_band_names):
        raise ValueError(
            f"Mismatch between number of soil bands ({soil_data.shape[0]}) "
            f"and soil band names ({len(soil_band_names)})"
        )

    feature_names = ['insar_cum', 'insar_diff'] + insar_aux_features + unique_vars + soil_band_names

    report.append("--- INPUT FILES ---")
    report.append(f"InSAR file: {insar_path}")
    report.append(f"ERA5 TIFF: {era5_path}")
    report.append(f"Soil TIFF: {soil_path}")
    report.append(f"ERA5 metadata TXT: {txt_metadata}\n")

    report.append("--- SOIL FEATURE CHECK ---")
    report.append(f"Soil TIFF shape: {soil_data.shape} (Bands, Rows, Cols)")
    report.append(f"Detected soil features: {', '.join(soil_band_names)}\n")

    # 2. Process InSAR HDF5
    print("Processing InSAR HDF5...")
    with h5py.File(insar_path, 'r') as f:
        i_rows, i_cols = f['vel'].shape
        zoom_factors = (e_rows / i_rows, e_cols / i_cols)

        resampled_mask = zoom(np.nan_to_num(f['mask'][:]), zoom_factors, order=1)
        valid_pixel_mask = resampled_mask > 0.90
        num_valid_pixels = np.sum(valid_pixel_mask)

        report.append("--- SPATIAL ALIGNMENT CHECK ---")
        report.append(f"Total ERA5 Grid Shape: ({e_rows}, {e_cols})")
        report.append(f"Soil Grid Shape: ({s_rows}, {s_cols})")
        report.append(f"Valid Clean Pixels (>90% coverage): {num_valid_pixels}")
        report.append(f"Discarded Edge Pixels: {(e_rows * e_cols) - num_valid_pixels}\n")

        dates_insar = f['imdates'][:].astype(str)
        months_insar = sorted(list(set([d[:6] for d in dates_insar])))
        full_timeline = sorted(list(set(df_era5['year'] + df_era5['month'])))

        # --- GAP HANDLING LOGIC ---
        print("Checking for temporal gaps...")
        consecutive_missing = 0
        cutoff_index = len(full_timeline)

        for i, month in enumerate(full_timeline):
            if month not in months_insar:
                consecutive_missing += 1
            else:
                consecutive_missing = 0

            if consecutive_missing > 2:
                print(f"Gap larger than 2 months detected at {month}. Truncating dataset.")
                cutoff_index = i - consecutive_missing + 1
                break

        # Update timeline based on cutoff
        full_timeline = full_timeline[:cutoff_index]
        missing_months = [m for m in full_timeline if m not in months_insar]

        report.append("--- TEMPORAL GAP ANALYSIS & FILLING ---")
        report.append(f"Timeline Truncated at index: {cutoff_index}")
        report.append(f"Final Timeline Length: {len(full_timeline)} months")
        report.append(f"Missing Months (to be filled): {len(missing_months)}")
        if missing_months:
            report.append(f"Interpolated months: {', '.join(missing_months)}")
        report.append("\n")

        # Resample Static InSAR Auxiliary Features
        print("Resampling static InSAR features...")
        resampled_aux = {}
        for aux in insar_aux_features:
            if aux == 'mask':
                resampled_aux[aux] = resampled_mask
            else:
                resampled_aux[aux] = zoom(np.nan_to_num(f[aux][:]), zoom_factors, order=1)

        # Resample Dynamic InSAR Cumulative Displacement
        cum_data = f['cum'][:]
        insar_data_mapped = {}
        for m in months_insar:
            if m in full_timeline:  # Only map months within our new truncated timeline
                indices = [i for i, d in enumerate(dates_insar) if d.startswith(m)]
                monthly_avg = np.nanmean(cum_data[indices], axis=0)
                insar_data_mapped[m] = zoom(np.nan_to_num(monthly_avg), zoom_factors, order=1)

    # 2.5 Prepare static soil features on valid pixels
    print("Preparing static soil features...")
    soil_features_matrix = soil_data.transpose(1, 2, 0)[valid_pixel_mask]  # (valid_pixels, soil_features)

    # 3. Temporal Interpolation (Linear for gaps <= 2)
    print("Interpolating temporal gaps...")
    interp_matrix = []
    for month in full_timeline:
        if month in insar_data_mapped:
            interp_matrix.append(insar_data_mapped[month][valid_pixel_mask])
        else:
            interp_matrix.append(np.full((num_valid_pixels,), np.nan))

    df_interp = pd.DataFrame(np.array(interp_matrix))
    df_interp = df_interp.interpolate(method='linear', axis=0, limit_direction='both')
    insar_cum_filled = df_interp.values

    # 3.5 Build month-to-month subsidence difference
    insar_diff_filled = np.zeros_like(insar_cum_filled)
    insar_diff_filled[0, :] = 0.0
    insar_diff_filled[1:, :] = insar_cum_filled[1:, :] - insar_cum_filled[:-1, :]

    # 4. Building Clean 3D Data Cube
    final_time_steps = []
    print("Building Filtered 3D Data Cube...")
    for i, month_str in enumerate(full_timeline):
        year, month = month_str[:4], month_str[4:]

        m_insar_cum = insar_cum_filled[i][:, np.newaxis]
        m_insar_diff = insar_diff_filled[i][:, np.newaxis]
        m_aux = np.stack([resampled_aux[a] for a in insar_aux_features]).transpose(1, 2, 0)[valid_pixel_mask]

        target_bands = df_era5[(df_era5['year'] == year) & (df_era5['month'] == month)]['idx'].values
        m_era5 = era5_data[target_bands - 1, :, :].transpose(1, 2, 0)[valid_pixel_mask]

        combined_month = np.hstack([m_insar_cum, m_insar_diff, m_aux, m_era5, soil_features_matrix])
        final_time_steps.append(combined_month)

    data_cube = np.stack(final_time_steps)
    final_lats = lats_full[valid_pixel_mask]
    final_lons = lons_full[valid_pixel_mask]

    # 5. Feature Ranges & Date Listing
    report.append("--- FEATURE NAME & RANGE SUMMARY (CLEAN & FILLED DATA) ---")
    report.append(f"{'Idx':<4} | {'Feature Name':<35} | {'Min':<15} | {'Max':<15}")
    report.append("-" * 80)
    for f_idx, f_name in enumerate(feature_names):
        f_min = np.nanmin(data_cube[:, :, f_idx])
        f_max = np.nanmax(data_cube[:, :, f_idx])
        report.append(f"{f_idx:<4} | {f_name:<35} | {f_min:<15.4f} | {f_max:<15.4f}")

    report.append(f"\n--- FINAL STRUCTURE ---")
    report.append(f"NPZ Shape: {data_cube.shape} (Time, Clean_Pixels, Features)")
    report.append(f"Time Range: {full_timeline[0]} to {full_timeline[-1]} ({len(full_timeline)} months)")
    report.append(f"Number of InSAR auxiliary features: {len(insar_aux_features)}")
    report.append(f"Number of ERA5 dynamic features: {len(unique_vars)}")
    report.append(f"Number of Soil static features: {len(soil_band_names)}")
    report.append(f"Total number of features: {len(feature_names)}")

    report.append("\n--- FEATURE BLOCK ORDER ---")
    report.append("1. InSAR dynamic features: insar_cum, insar_diff")
    report.append(f"2. InSAR auxiliary features: {', '.join(insar_aux_features)}")
    report.append(f"3. ERA5 monthly dynamic features: {', '.join(unique_vars)}")
    report.append(f"4. Soil static features: {', '.join(soil_band_names)}")

    # 6. Save Files
    print(f"Saving merged data to {output_npz}...")
    np.savez_compressed(
        output_npz,
        data=data_cube,
        lats=final_lats,
        lons=final_lons,
        features=np.array(feature_names, dtype=object),
        dates=np.array(full_timeline, dtype=object)
    )

    with open(output_report, 'w', encoding='utf-8') as fr:
        fr.write("\n".join(report))

    print("\n--- Process Complete ---")
    print(f"Final Data Shape: {data_cube.shape}")
    print(f"Total Features: {len(feature_names)}")
    print(f"Soil Features Added: {soil_band_names}")
    print(f"Detailed report saved to: {output_report}")


if __name__ == "__main__":
    align_and_save()

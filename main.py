import nd2_processing_class
import single_position_data_class
import position_analysis_class


if __name__ == "__main__":
    nd2_path = r"Z:\Users\CHENDAV\nikon_confocal\Calcin_Dextran_CAAX_RFP_20250410\Calcin_Dextran_CAAX_RFP_20250414.nd2"

    # You can control channels and frames like this:
    processor = ND2Processing(
        nd2_path,
        channels=[1, 2],        # or None for all channels
        frames_to_load=15       # or None for all timepoints
    )
    
    processor.process_fovs()
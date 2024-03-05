import pandas as pd

input_file_path = 'US_youtube_trending_data.csv'
output_file_path = 'clean_data.csv'
chunksize = 1000  # Use a small chunksize to manage memory usage

# No need to initialize an empty DataFrame when working directly with CSV
# df_clean = None

# Open the output file in append mode, so we can keep adding chunks to it
with open(output_file_path, 'a', newline='', encoding='utf-8') as f:
    # Initialize a flag to indicate whether we need to write headers
    write_header = True
    try:
        # Iterate over the file in chunks
        for chunk in pd.read_csv(input_file_path, chunksize=chunksize, error_bad_lines=False):
            # Write the chunk to CSV with or without the header
            chunk.to_csv(f, header=write_header, index=False, mode='a')
            # After the first chunk is written, set write_header to False
            write_header = False

    except pd.errors.ParserError as e:
        print(f"Encountered an error: {e}")
        # This exception block can be used to log errors or handle them as needed

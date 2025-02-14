import os
import laspy

# Define input and output directories
input_dir = "uavlidar/original_las"
output_dir = os.path.join(input_dir, "compressed")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all .las files and compress them to .laz
for file in os.listdir(input_dir):
    if file.endswith(".las"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".las", ".laz"))

        try:
            print(f"Processing {file}...")

            # Open the LAS file in read mode
            with laspy.open(input_path) as las_file:
                header = las_file.header
                point_format = header.point_format

                # Create a new LAZ file with the same header
                with laspy.open(output_path, mode="w", header=header, do_compress=True) as laz_file:
                    for points in las_file.chunk_iterator(10_000_000):  # Read in chunks
                        laz_file.write_points(points)

            print(f"✔ Successfully compressed: {file} → {output_path}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

print("Compression process completed.")

#!/bin/bash

# Script to copy 40% of already extracted SAM embeddings from each class folder
# maintaining the stratified sampling ratio

SOURCE_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/sam_cache_unified"
DEST_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/sam_cache_unified_40"
IMAGE_DIR="/scratch/gilbreth/abelde/Thesis/StructureAwareGen/dataset/imagenet-1K-hf/train"

echo "Creating destination directory structure..."
mkdir -p "$DEST_DIR"

# Iterate through each class folder in the image directory
for class_folder in "$IMAGE_DIR"/*; do
    if [ -d "$class_folder" ]; then
        class_id=$(basename "$class_folder")
        
        # Count total images in this class
        total_images=$(find "$class_folder" -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
        
        # Calculate 40% (round down)
        target_count=$((total_images * 40 / 100))
        
        echo "Processing class $class_id: $total_images images → keeping $target_count (40%)"
        
        # Check if source class folder exists
        if [ ! -d "$SOURCE_DIR/$class_id" ]; then
            echo "  ⚠️  Warning: No embeddings found for class $class_id, skipping..."
            continue
        fi
        
        # Create destination subdirectories
        mkdir -p "$DEST_DIR/$class_id/masks_npz"
        mkdir -p "$DEST_DIR/$class_id/meta"
        
        # Get list of all .npz files in source, sort them, take first 40%
        npz_files=($(find "$SOURCE_DIR/$class_id/masks_npz" -name "*.npz" | sort))
        actual_count=${#npz_files[@]}
        
        # Recalculate target based on actual extracted files
        if [ $actual_count -gt 0 ]; then
            copy_count=$((actual_count * 40 / 100))
            
            echo "  Found $actual_count extracted, copying $copy_count files..."
            
            # Copy first 40% of .npz files
            for ((i=0; i<copy_count; i++)); do
                npz_file="${npz_files[$i]}"
                filename=$(basename "$npz_file")
                
                # Copy .npz file
                cp "$npz_file" "$DEST_DIR/$class_id/masks_npz/"
                
                # Copy corresponding .json file (replace .npz with .json)
                json_filename="${filename%.npz}.json"
                json_file="$SOURCE_DIR/$class_id/meta/$json_filename"
                
                if [ -f "$json_file" ]; then
                    cp "$json_file" "$DEST_DIR/$class_id/meta/"
                fi
            done
            
            echo "  ✓ Copied $copy_count files for class $class_id"
        else
            echo "  ⚠️  No extracted files found for class $class_id"
        fi
    fi
done

echo ""
echo "Summary:"
echo "========"
total_npz=$(find "$DEST_DIR" -name "*.npz" | wc -l)
total_json=$(find "$DEST_DIR" -name "*.json" | wc -l)
total_classes=$(find "$DEST_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "Total classes processed: $total_classes"
echo "Total .npz files copied: $total_npz"
echo "Total .json files copied: $total_json"
echo ""
echo "✓ 40% stratified subset created at: $DEST_DIR"
echo ""
echo "Next steps:"
echo "1. Update your bash script to use: --output_dir $DEST_DIR"
echo "2. Run with --skip_existing to only extract remaining 40% classes"
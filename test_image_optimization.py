#!/usr/bin/env python3
"""
Test script for image optimization module
Demonstrates usage and shows optimization results
"""

import os
import sys
from pathlib import Path
from image_optimizer import (
    optimize_image,
    ImageOptimizer,
    TemplateImageOptimizer,
    ThumbnailOptimizer,
    CoverImageOptimizer
)


def format_size(size_bytes):
    """Format byte size to human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def test_single_image(image_path):
    """Test optimization on a single image with different optimizers"""
    
    print(f"\n{'='*70}")
    print(f"Testing Image Optimization")
    print(f"{'='*70}")
    print(f"\nImage: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return
    
    # Read original image
    with open(image_path, 'rb') as f:
        original_data = f.read()
    
    original_size = len(original_data)
    print(f"\nOriginal Size: {format_size(original_size)}")
    
    # Test different optimizers
    optimizers = [
        ("Quick Optimize (Default)", lambda: optimize_image(original_data)),
        ("Template Optimizer", lambda: TemplateImageOptimizer().optimize_image(original_data)),
        ("Cover Optimizer", lambda: CoverImageOptimizer().optimize_image(original_data)),
        ("Thumbnail Optimizer", lambda: ThumbnailOptimizer().optimize_image(original_data)),
        ("JPEG Format", lambda: optimize_image(original_data, format="JPEG")),
        ("Lower Quality WebP", lambda: optimize_image(original_data, quality=75)),
    ]
    
    print(f"\n{'Optimizer':<30} {'Size':<15} {'Savings':<12} {'Format':<10}")
    print("-" * 70)
    
    results = []
    for name, optimizer_fn in optimizers:
        try:
            optimized, content_type, ext = optimizer_fn()
            optimized_size = len(optimized)
            savings = (1 - optimized_size / original_size) * 100
            
            print(f"{name:<30} {format_size(optimized_size):<15} {savings:>6.1f}%     {ext:<10}")
            
            results.append({
                'name': name,
                'data': optimized,
                'ext': ext,
                'size': optimized_size
            })
        except Exception as e:
            print(f"{name:<30} ❌ Failed: {str(e)[:30]}")
    
    # Offer to save optimized versions
    print(f"\n{'='*70}")
    print("Save optimized versions? (y/n): ", end='')
    
    try:
        response = input().strip().lower()
        if response == 'y':
            save_optimized_versions(image_path, results)
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping save.")


def save_optimized_versions(original_path, results):
    """Save optimized versions to disk"""
    
    # Create output directory
    base_path = Path(original_path).parent
    output_dir = base_path / "optimized_output"
    output_dir.mkdir(exist_ok=True)
    
    original_name = Path(original_path).stem
    
    saved_count = 0
    for result in results:
        # Clean name for filename
        clean_name = result['name'].replace(' ', '_').replace('(', '').replace(')', '').lower()
        output_path = output_dir / f"{original_name}_{clean_name}.{result['ext']}"
        
        with open(output_path, 'wb') as f:
            f.write(result['data'])
        
        print(f"✅ Saved: {output_path}")
        saved_count += 1
    
    print(f"\n✅ Saved {saved_count} optimized versions to: {output_dir}")


def test_batch(directory):
    """Test optimization on all images in a directory"""
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
    
    print(f"\n{'='*70}")
    print(f"Batch Testing Image Optimization")
    print(f"{'='*70}")
    print(f"\nDirectory: {directory}")
    
    if not os.path.isdir(directory):
        print(f"❌ Error: Directory not found: {directory}")
        return
    
    # Find all images
    image_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No image files found in {directory}")
        return
    
    print(f"\nFound {len(image_files)} image(s)")
    print(f"\n{'File':<30} {'Original':<15} {'Optimized':<15} {'Savings':<12}")
    print("-" * 75)
    
    total_original = 0
    total_optimized = 0
    
    for image_path in image_files:
        filename = Path(image_path).name
        
        try:
            with open(image_path, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            optimized, _, _ = optimize_image(original_data)
            optimized_size = len(optimized)
            
            savings = (1 - optimized_size / original_size) * 100
            
            print(f"{filename[:28]:<30} {format_size(original_size):<15} {format_size(optimized_size):<15} {savings:>6.1f}%")
            
            total_original += original_size
            total_optimized += optimized_size
            
        except Exception as e:
            print(f"{filename[:28]:<30} ❌ Failed: {str(e)[:30]}")
    
    print("-" * 75)
    
    if total_original > 0:
        total_savings = (1 - total_optimized / total_original) * 100
        print(f"{'TOTAL':<30} {format_size(total_original):<15} {format_size(total_optimized):<15} {total_savings:>6.1f}%")


def interactive_mode():
    """Interactive mode for testing"""
    
    print("\n" + "="*70)
    print(" Image Optimization Test Suite")
    print("="*70)
    print("\nOptions:")
    print("  1. Test single image")
    print("  2. Batch test directory")
    print("  3. Exit")
    
    while True:
        print("\nSelect option (1-3): ", end='')
        try:
            choice = input().strip()
            
            if choice == '1':
                print("\nEnter image path: ", end='')
                path = input().strip()
                test_single_image(path)
            
            elif choice == '2':
                print("\nEnter directory path: ", end='')
                path = input().strip()
                test_batch(path)
            
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option. Please select 1-3.")
        
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        # Command line mode
        path = sys.argv[1]
        
        if os.path.isfile(path):
            test_single_image(path)
        elif os.path.isdir(path):
            test_batch(path)
        else:
            print(f"❌ Error: Path not found: {path}")
            print("\nUsage:")
            print("  python test_image_optimization.py <image_file>")
            print("  python test_image_optimization.py <directory>")
            print("  python test_image_optimization.py  (interactive mode)")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

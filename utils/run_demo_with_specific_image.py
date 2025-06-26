#!/usr/bin/env python3
"""
Run the demo visualization with the specific image requested by the user
"""

import subprocess
import sys
from pathlib import Path

def run_demo_with_specific_image():
    """Run the demo with the specific image: 3babc2cec15ad848.jpg"""
    
    # Define the image path
    image_path = "/root/data/root/23tnt/Track1/retrieval/Input/3babc2cec15ad848.jpg"
    output_path = "demo_3babc2cec15ad848.png"
    
    print("🚀 Running Event-Enriched Image Retrieval and Captioning Demo")
    print("=" * 70)
    print(f"📸 Using image: {image_path}")
    print(f"💾 Output will be saved to: {output_path}")
    print()
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found at {image_path}")
        print("💡 Please check the file path and try again.")
        return False
    
    # Check if demo script exists
    demo_script = Path("demo_visualization.py")
    if not demo_script.exists():
        print(f"❌ Error: Demo script not found: {demo_script}")
        print("💡 Make sure you're in the correct directory.")
        return False
    
    print("🔧 Running demo visualization...")
    print("-" * 50)
    
    # Run the demo
    cmd = [
        "python", 
        "demo_visualization.py",
        "--image_path", image_path,
        "--output_path", output_path,
        "--llava_model_id", "llava-hf/llava-v1.6-mistral-7b-hf"
    ]
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✅ Demo completed successfully!")
            print(f"🖼️ Check the output file: {output_path}")
            print("📊 The visualization shows:")
            print("   • Input query image")
            print("   • Retrieved similar articles")
            print("   • Article content used for captioning")
            print("   • Generated event-enriched caption")
            return True
        else:
            print(f"\n❌ Demo failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out (took longer than 1 hour)")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def main():
    """Main function"""
    
    # Change to the correct directory
    target_dir = Path("/root/data/root/23tnt/Track1/retrieval")
    if target_dir.exists():
        import os
        os.chdir(target_dir)
        print(f"📁 Changed to directory: {target_dir}")
    else:
        print(f"❌ Target directory not found: {target_dir}")
        return
    
    # Run the demo
    success = run_demo_with_specific_image()
    
    if success:
        print("\n🎉 Demo visualization completed successfully!")
        print("📝 The generated image shows the complete pipeline:")
        print("   1. Your input image (3babc2cec15ad848.jpg)")
        print("   2. Retrieved similar articles from the database")
        print("   3. Article content used for context")
        print("   4. Generated event-enriched caption")
    else:
        print("\n⚠️ Demo encountered issues. Please check the error messages above.")
        print("💡 Common solutions:")
        print("   • Make sure all required packages are installed")
        print("   • Check if CUDA is available for GPU acceleration")
        print("   • Verify database files exist and are accessible")

if __name__ == "__main__":
    main() 
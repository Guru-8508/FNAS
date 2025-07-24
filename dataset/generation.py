import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_thermal_dataset(num_samples=2000, output_file="thermal_face_dataset.csv"):
    """
    Generate synthetic thermal face dataset for health deficiency detection
    Compatible with TensorFlow 2.13.0 and NumPy 1.24.3
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    print(f"Generating {num_samples} thermal face samples...")
    
    # Initialize data storage
    thermal_data = []
    
    for i in range(num_samples):
        # Base physiological temperature (36.5°C ± 0.8°C)
        base_temp = np.random.normal(36.5, 0.8)
        
        # Regional temperature variations
        forehead_temp = base_temp + np.random.normal(0.5, 0.4)    # Warmer region
        eye_temp = base_temp + np.random.normal(0.2, 0.3)         # Slightly warm
        cheek_temp = base_temp + np.random.normal(0.0, 0.3)       # Baseline
        nose_temp = base_temp + np.random.normal(-0.3, 0.3)       # Cooler region
        chin_temp = base_temp + np.random.normal(-0.1, 0.3)       # Slightly cool
        
        # Calculate aggregate measurements
        all_temps = [forehead_temp, eye_temp, cheek_temp, nose_temp, chin_temp]
        avg_temp = np.mean(all_temps)
        max_temp = max(all_temps) + np.random.normal(0.5, 0.2)
        min_temp = min(all_temps) - np.random.normal(0.5, 0.2)
        temp_std = np.std(all_temps)
        
        # Ensure realistic bounds (32-42°C)
        temps = [avg_temp, max_temp, min_temp, forehead_temp, 
                eye_temp, cheek_temp, nose_temp, chin_temp, temp_std]
        temps = [np.clip(temp, 32.0, 42.0) for temp in temps]
        
        # Health deficiency detection logic
        deficiency_score = 0.0
        
        # Temperature anomaly indicators
        if avg_temp < 35.5 or avg_temp > 38.5:  # Abnormal average
            deficiency_score += 0.3
        if max_temp - min_temp > 3.5:  # High variation
            deficiency_score += 0.25
        if forehead_temp - nose_temp > 2.5:  # Regional imbalance
            deficiency_score += 0.2
        if temp_std > 1.2:  # High standard deviation
            deficiency_score += 0.15
        
        # Binary classification (30% deficiency rate)
        deficiency_label = 1 if np.random.random() < (0.3 + deficiency_score * 0.5) else 0
        
        # Store sample
        sample = {
            'avg_temp': round(temps[0], 2),
            'max_temp': round(temps[1], 2),
            'min_temp': round(temps[2], 2),
            'forehead_temp': round(temps[3], 2),
            'eye_temp': round(temps[4], 2),
            'cheek_temp': round(temps[5], 2),
            'nose_temp': round(temps[6], 2),
            'chin_temp': round(temps[7], 2),
            'temp_std': round(temps[8], 2),
            'deficiency_label': deficiency_label
        }
        thermal_data.append(sample)
        
        # Progress tracking
        if (i + 1) % 500 == 0:
            print(f"Progress: {i + 1}/{num_samples} samples generated")
    
    # Create DataFrame
    df = pd.DataFrame(thermal_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Dataset summary
    healthy_count = len(df[df['deficiency_label'] == 0])
    deficiency_count = len(df[df['deficiency_label'] == 1])
    
    print(f"\n✅ Dataset Generation Complete!")
    print(f"📁 File: {output_file}")
    print(f"📊 Total samples: {len(df)}")
    print(f"🟢 Healthy: {healthy_count} ({healthy_count/len(df)*100:.1f}%)")
    print(f"🔴 Deficiency: {deficiency_count} ({deficiency_count/len(df)*100:.1f}%)")
    print(f"🌡️ Temperature range: {df['avg_temp'].min():.1f}°C - {df['avg_temp'].max():.1f}°C")
    
    return df

# Generate the dataset
dataset = generate_thermal_dataset(2000, "thermal_face_dataset.csv")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("PLASTIC-DEGRADING ENZYME CLASSIFIER - QUICK DEMO")
print("=" * 70)

print("\n[1/4] Loading pre-trained PET classification model...")
model = joblib.load('generated_models/PET_clf.joblib')
print(f"✓ Model loaded: {type(model).__name__}")

print("\n[2/4] Loading sample enzyme sequences...")
data = pd.read_csv('raw_data/data_sequences.csv')
print(f"✓ Dataset loaded: {len(data)} sequences available")
print(f"✓ Columns: {list(data.columns)}")

sample_sequences = data.head(5)
print(f"\n[3/4] Sample sequences:")
for idx, row in sample_sequences.iterrows():
    seq = row['sequence']
    seq_id = row.get('id', f'seq_{idx}')
    print(f"  • {seq_id}: {seq[:50]}..." if len(seq) > 50 else f"  • {seq_id}: {seq}")

print("\n[4/4] Loading encoded sequences for prediction...")
print("   Using ProtTrans-XLU50 embeddings (matches trained model)...")
try:
    encoded_data = pd.read_csv('processed_dataset/protrans_xlu50/encoder_data.csv')
    print(f"✓ Encoded data loaded: {encoded_data.shape}")
    
    if 'sequence' in encoded_data.columns:
        sequences_from_encoded = encoded_data['sequence'].head(5)
        encoded_data = encoded_data.drop('sequence', axis=1)
        print(f"✓ Removed sequence column, keeping only numerical embeddings")
    
    print(f"✓ Features: {encoded_data.shape[1]} embedding dimensions")
    
    sample_encoded = encoded_data.head(5)
    
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)
    print("Using ProtTrans-XLU50 protein language model embeddings as input...\n")
    
    predictions = model.predict(sample_encoded)
    probabilities = model.predict_proba(sample_encoded)
    
    print(f"\nPredicting if enzymes can degrade PET plastic:\n")
    
    for i, (idx, row) in enumerate(sample_sequences.iterrows()):
        seq_id = row.get('id', f'seq_{i}')
        pred = predictions[i]
        prob = probabilities[i]
        
        result = "✓ CAN DEGRADE PET" if pred == 1 else "✗ Cannot degrade PET"
        confidence = prob[1] if pred == 1 else prob[0]
        
        print(f"Sequence {i+1} ({seq_id}):")
        print(f"  Prediction: {result}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Probabilities: [No: {prob[0]:.2%}, Yes: {prob[1]:.2%}]")
        print()
    
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nAvailable models for other plastics:")
    import os
    models = [f for f in os.listdir('generated_models') if f.endswith('.joblib')]
    for model_file in sorted(models):
        plastic = model_file.replace('_clf.joblib', '')
        print(f"  • {plastic}")
    
    print("\nTo use a different model, change:")
    print("  model = joblib.load('generated_models/PLA_clf.joblib')  # For PLA")
    print("  model = joblib.load('generated_models/NYLON_PA_clf.joblib')  # For Nylon")
    
except FileNotFoundError as e:
    print(f"\n⚠ Error: Could not find encoded data file")
    print(f"   {e}")
    print("\n💡 You may need to run encoding notebooks first:")
    print("   jupyter notebook src/encoding_approaches/embedding_strategies.ipynb")
except Exception as e:
    print(f"\n⚠ Error during prediction: {e}")
    print("   Check that the encoded data matches the model's expected input format")

import joblib
import pandas as pd

print("=" * 70)
print("TESTING IDEONELLA SAKAIENSIS PETASE")
print("=" * 70)

target_sequence = """MNFPRKLRAGLALALALPALAAPAAADKAEAVTLDDFDKG
GIVVGRNTSDTLYEGLKDRLTVSSTANVTKLGDLGGLD
NTQVIVPGFNATGFGYGQSTGAKADATFKAEKAKKAV
LAAGADVVVVATGGGTSFGANLARQLGADVVVTAGAG
ASAGANFTRAFEGVTPTRLGAVNQVVASGGTSLFGANN
TRAFEGVTPTRLGAVNQVVASGGTSLFGANNTRAFEGV
TPTRLGAVNQVVASGGTSLFGANNTRAFEGVTPTRLG"""

target_sequence = target_sequence.replace('\n', '').replace(' ', '')

print(f"\nTarget sequence:")
print(f"  ID: sp|A0A0K8P6T7|PETASE_IDESA")
print(f"  Name: PETase from Ideonella sakaiensis")
print(f"  Length: {len(target_sequence)} amino acids")
print(f"  Sequence: {target_sequence[:60]}...")

print("\n[1/3] Loading dataset to check if sequence exists...")
data = pd.read_csv('raw_data/data_sequences.csv')
print(f"✓ Dataset loaded: {len(data)} sequences")

matching_idx = data[data['sequence'] == target_sequence].index
if len(matching_idx) > 0:
    idx = matching_idx[0]
    print(f"✓ FOUND! This sequence is at index {idx} in the dataset")
else:
    print("✗ This sequence is NOT in the pre-computed dataset")
    print("\n💡 To test new sequences, you need to:")
    print("   1. Add the sequence to raw_data/data_sequences.csv")
    print("   2. Run encoding notebook: src/encoding_approaches/embedding_strategies.ipynb")
    print("   3. Then use the trained model for prediction")
    print("\nAlternatively, checking similar sequences in the dataset...")
    
    for i, row in data.head(10).iterrows():
        seq = row['sequence']
        if 'PETASE' in seq.upper() or len(seq) == len(target_sequence):
            print(f"  - Sequence {i}: length {len(seq)}")
    exit()

print("\n[2/3] Loading pre-trained PET classification model...")
model = joblib.load('generated_models/PET_clf.joblib')
print(f"✓ Model loaded: {type(model).__name__}")

print("\n[3/3] Loading encoded sequence and making prediction...")
encoded_data = pd.read_csv('processed_dataset/protrans_xlu50/encoder_data.csv')

if 'sequence' in encoded_data.columns:
    encoded_data = encoded_data.drop('sequence', axis=1)

encoded_sequence = encoded_data.iloc[idx:idx+1]
print(f"✓ Encoded sequence loaded: {encoded_sequence.shape}")

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

prediction = model.predict(encoded_sequence)[0]
probabilities = model.predict_proba(encoded_sequence)[0]

result = "✓ CAN DEGRADE PET" if prediction == 1 else "✗ CANNOT degrade PET"
confidence = probabilities[1] if prediction == 1 else probabilities[0]

print(f"\nSequence: sp|A0A0K8P6T7|PETASE_IDESA")
print(f"Organism: Ideonella sakaiensis")
print(f"")
print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2%}")
print(f"Probabilities:")
print(f"  - Cannot degrade PET: {probabilities[0]:.2%}")
print(f"  - CAN degrade PET:    {probabilities[1]:.2%}")

print("\n" + "=" * 70)
if prediction == 1:
    print("✓ SUCCESS! The model correctly identifies this as a PET-degrading enzyme.")
    print("  This is the famous PETase discovered in 2016!")
else:
    print("⚠ The model predicted this cannot degrade PET.")
    print("  This might be unexpected as this is a known PET-degrading enzyme.")
print("=" * 70)

"""Quick diagnostic to find training issues."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)

# Fix Windows unicode issues
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("DIAGNOSTIC TEST")
print("=" * 60)

try:
    print("\n[1/6] Importing modules...")
    from app.data.loader import DataLoader
    from app.data.preprocessor import Preprocessor, TechnicalIndicatorEngine
    from app.models.fusion import NeuroEconometricNet
    import torch
    print("[OK] All imports successful")
    
    print("\n[2/6] Loading data...")
    dl = DataLoader(ticker='^GSPC', local_csv='data/GSPC_ohlcv.csv')
    ohlcv_df, news_df = dl.load_all('2023-01-01', '2024-12-31', include_news=False)
    print(f"[OK] Loaded {len(ohlcv_df)} rows")
    
    print("\n[3/6] Computing technical indicators...")
    df_with_indicators = TechnicalIndicatorEngine.compute_all(ohlcv_df)
    print(f"[OK] Added {len(df_with_indicators.columns) - len(ohlcv_df.columns)} indicators")
    
    print("\n[4/6] Preprocessing...")
    preprocessor = Preprocessor()
    processed_df, metadata = preprocessor.fit_transform(df_with_indicators)
    print(f"[OK] Processed shape: {processed_df.shape}")
    print(f"[OK] Metadata keys: {list(metadata.keys())}")
    
    print("\n[5/6] Creating neural network...")
    model = NeuroEconometricNet(
        input_dim=processed_df.shape[1] - 1,  # Exclude target
        hidden_dim=64,
        nhead=4,
        num_lstm_layers=2,
        dropout=0.2
    )
    print(f"[OK] Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    print("\n[6/6] Testing forward pass...")
    # Create dummy batch
    batch_size = 2
    seq_len = 20
    dummy_prices = torch.randn(batch_size, seq_len, processed_df.shape[1] - 1)
    dummy_ardl = torch.randn(batch_size)
    dummy_vol = torch.randn(batch_size)
    dummy_sent = torch.randn(batch_size)
    
    output = model(dummy_prices, dummy_ardl, dummy_vol, dummy_sent)
    print(f"[OK] Forward pass successful. Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()

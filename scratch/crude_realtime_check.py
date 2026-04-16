from core.inference import InferenceEngine
from core.regime_detector import MarketRegime
import logging

# Set up logging to stdout to see the trace
logging.basicConfig(level=logging.INFO)

def diagnose_crude():
    engine = InferenceEngine()
    print("\n--- DIAGNOSING CRUDEOIL REGIME & SIGNAL ---")
    
    # Run prediction exactly like the terminal/sentinel does
    result = engine.predict_symbol("CrudeOIL", save_to_db=False)
    
    if result:
        print(f"Symbol: {result['symbol']}")
        print(f"Regime: {result['regime']}")
        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Tradeable: {result.get('is_authorized', False)}")
        print(f"Hidden: {result.get('is_hidden', False)}")
        
        if result['regime'] == 'CRISIS':
            print("!!! CRISIS DETECTED !!!")
        else:
            print(f"Regime is {result['regime']}, matching database logs.")
    else:
        print("No result found for CrudeOIL.")

if __name__ == "__main__":
    diagnose_crude()

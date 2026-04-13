
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.specialist_factory import SpecialistFactory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Factory
factory = SpecialistFactory(
    min_win_rate=0.70,
    min_samples=1000
)

print("🚀 Starting Test Training: AUDCAD (SELL)")
# Train only SELL
success = factory.train_specialist("AUDCAD", "SELL")

if success:
    print("\n✅ AUDCAD SELL Certified!")
else:
    print("\n❌ AUDCAD SELL Failed.")

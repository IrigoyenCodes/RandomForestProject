# test_import.py
from randomForest2 import NBAPredictorModel

def test():
    try:
        model = NBAPredictorModel()
        print("SUCCESS! NBAPredictorModel imported correctly")
        print(f"Class location: {NBAPredictorModel.__module__}")
    except Exception as e:
        print(f"IMPORT FAILED: {str(e)}")

if __name__ == "__main__":
    test()
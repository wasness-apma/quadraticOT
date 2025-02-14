from .require import require

def test_require():
    try:
        require(False)
    except:
        require(True)
    else:
        raise Exception("Require(false) should throw")

def run_all_tests():
    test_require()
    print("Ran all tests for core_testing.py")

if __name__ == "__main__":
    run_all_tests()
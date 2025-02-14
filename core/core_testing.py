from .require import require, requireApproxEq

def test_require():
    try:
        require(False)
    except:
        require(True)
    else:
        raise Exception("Require(false) should throw")

def test_require_approx():
    try:
        requireApproxEq(1.0, 2.0, 0.1)
    except:
        requireApproxEq(1.0, 2.0, 3.0)
    else:
        raise Exception("Require(false) should throw")

def run_all_tests():
    test_require()
    test_require_approx()
    print("Ran all tests for core_testing.py")

if __name__ == "__main__":
    run_all_tests()
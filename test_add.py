from function import Function
from add import add

def test_add():
    f1 = Function([0, 1, 2, 3], Ts=1, f=[1, 2, 3, 4])
    f2 = Function([0, 1, 2, 3], Ts=1, f=[4, 3, 2, 1])
    f3 = add(f1, f2)
    assert f3.f == [5, 5, 5, 5], "Test case 1 failed"

def test_different_length():
    f1 = Function([0, 1, 2, 3, 4], Ts=1, f=[1, 2, 3, 4, 1])
    f2 = Function([0, 1, 2, 3], Ts=1, f=[4, 3, 2, 1])
    f3 = add(f1, f2)
    assert f3.f == [5, 5, 5, 5, 1], "Test case 2 failed"

def test_different_sampling_time():
    f1 = Function([0, 1, 2, 3], Ts=1, f=[1, 2, 3, 4])
    f2 = Function([0, 1, 2, 3], Ts=2, f=[4, 3, 2, 1])
    try:
        f3 = add(f1, f2)
    except ValueError:
        assert True
    else:
        assert False, "Test case 3 failed"

def test_empty_function():
    f1 = Function([0, 1, 2, 3], Ts=1, f=[1, 2, 3, 4])
    f2 = Function([], Ts=1, f=[])
    f3 = add(f1, f2)
    assert f3.f == [1, 2, 3, 4], "Test case 4 failed"

def test_cummutative():
    f1 = Function([0, 1, 2, 3], Ts=1, f=[1, 2, 3, 4])
    f2 = Function([0, 1, 2, 3], Ts=1, f=[4, 3, 2, 1])
    f3 = add(f1, f2)
    f4 = add(f2, f1)
    assert f3.f == f4.f, "Test case 5 failed"

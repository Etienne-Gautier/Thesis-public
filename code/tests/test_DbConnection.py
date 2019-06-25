import unittest
from tests.context import DbAccess
from DbAccess.DbConnect import DbConnect

class DbConnectTest(unittest.TestCase):
    def test_connection(self):
        connector = DbConnect()
        self.assertTrue(True) #if this line is executed, the connection is created a database named Thesis exists and contains the KITTI collection
    


if __name__ == '__main__':
    unittest.main()
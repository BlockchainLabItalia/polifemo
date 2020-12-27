from influxdb import InfluxDBClient, DataFrameClient

class MyInflux:
    def __init__(self, host, database_name, port):
        self.db_name = database_name
        self.client = InfluxDBClient(host, port)
        if(not database_name in self.get_databases()):
            self.client.create_database(database_name)
        self.client.switch_database(database_name)
        
    def get_databases(self):
        dbs = self.client.get_list_database()
        l = [i['name'] for i in dbs]
        return l

    def write_revealed(self, peopleIN, peopleOUT, cameraID):
        self.client.write_points([{
            "measurement": "revealed",
            "tags": { "camera": cameraID },
            "fields": { "people_in": peopleIN, "people_out": peopleOUT },
        }])

    def write_crossed(self, goingIN, goingOUT, cameraID):
        self.client.write_points([{
            "measurement": "crossed",
            "tags": { "camera": cameraID },
            "fields": { "people_going_in": goingIN, "people_going_out": goingOUT },
            
        }])
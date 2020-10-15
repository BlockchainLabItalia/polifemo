from influxdb import InfluxDBClient, DataFrameClient

class MyInflux:
    def __init__(self, cameraID, database_name='socialcounter'):
        self.db_name = database_name
        self.client = InfluxDBClient(host='68.183.38.198', port=8086)
        if(not database_name in self.get_databases()):
            self.client.create_database(database_name)

        self.client.switch_database(database_name)
        self.cameraId = cameraID
        
    def get_databases(self):
        dbs = self.client.get_list_database()
        l = [i['name'] for i in dbs]
        return l

    def write_revealed(self, peopleIN, peopleOUT):
        self.client.write_points([{
            "measurement": "revealed",
            "tags": { "camera": self.cameraId },
            "fields": { "people_in": peopleIN, "people_out": peopleOUT },
        }])

    def write_crossed(self, goingIN, goingOUT):
        self.client.write_points([{
            "measurement": "crossed",
            "tags": { "camera": self.cameraId },
            "fields": { "people_going_in": goingIN, "people_going_out": goingOUT },
            
        }])
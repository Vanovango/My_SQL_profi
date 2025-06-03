import mysql.connector

try:
    connection = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        passwd="2431",
        database="reg_finance",
        connect_timeout=60
    )
    print("Successful connection")

except Exception as error:
    print("Connection failed")
    print(error)
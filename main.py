# import matplotlib.pyplot as plt
import mysql.connector
import networkx as nx
import sqlalchemy
from flask import Flask, jsonify, request
# from flask_cors import CORS
from networkx import Graph
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = sqlalchemy.orm.declarative_base()


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    dp_url = Column(String, nullable=False)


# Specify the database driver in the connection string (mysql+mysqlconnector)
engine = create_engine("mysql+mysqlconnector://root:admin@localhost/mydb")

# Create the tables within a try-except block to catch potential exceptions
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"An error occurred while creating database tables: {e}")

# Set up a session
Session = sessionmaker(bind=engine)
session = Session()

app = Flask(__name__)
# CORS(app)
# CORS(app, origins=["http://localhost:3306/"])
graph = Graph()

cnx = mysql.connector.connect(user='root', password='root', host='127.0.0.1', port='3306', database='mydb')
cursor = cnx.cursor()
cnx.close()

Base = declarative_base()


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name = data["name"]
    username = data["username"]
    email = data["email"]
    password = data["password"]
    dp_url = data["dp_url"]
    user = session.query(User).filter_by(username=username).first()
    if user:
        return jsonify({"message": f"User {username} already exists in the network", "status": 409}), 409
    # insert the user data into the MySQL database
    add_user = "INSERT INTO user (name, username, email, password,dp_url) VALUES (%s, %s, %s, %s)"
    cursor.execute(add_user, (name, username, email, password, dp_url))
    cnx.commit()
    graph.add_node(username, name=name, email=email)
    return jsonify({"message": f"User {name} added to the network", "status": 201}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data["username"]
    password = data["password"]
    # retrieve the user data from the MySQL database
    get_user = ("SELECT name, email FROM user WHERE username = %s and password = %s")
    cursor.execute(get_user, (username, password))
    user = cursor.fetchone()
    if user is None:
        return jsonify({"message": "Invalid username or password", "status": 401}), 401
    name, email = user
    return jsonify({"message": f"Welcome {name}", "status": 200}), 200


@app.route("/add_user", methods=["POST"])
def add_user():
    data = request.get_json()
    name = data["name"]
    username = data["username"]
    email = data["email"]
    password = data.get("password", "samsung")
    dp_url = data.get("dp_url")
    user = session.query(User).filter_by(username=username).first()
    if user:
        return jsonify({"message": f"User {username} already exists in the network", "status": 409}), 409
    try:
        new_user = User(name=name, username=username, email=email, password=password, dp_url=dp_url)
        session.add(new_user)
        session.commit()
        graph.add_node(username, name=name, email=email)
        return jsonify({"message": f"User {name} added to the network", "status": 201}), 201
    except Exception as e:
        session.rollback()
        print(f"Error adding user: {e}")
        return jsonify({"message": "Error adding user to the database", "status": 500}), 500


@app.route("/view_profile/<username>", methods=["GET"])
def view_profile(username):
    if not graph.has_node(username):
        return jsonify({"message": f"User {username} does not exist in the network", "status": 404}), 404
    details = nx.get_node_attributes(graph, username)
    return jsonify({"username": username, "details": details, "status": 200}), 200


# @app.route("/add_friend", methods=["POST"])
# def add_friend():
#     data = request.get_json()
#     user = data["user"]
#     friend = data["friend"]
#     try:
#         cnx = mysql.connector.connect(user='root', password='admin', host='localhost', database='mydb')
#         cursor = cnx.cursor()
#         query = "SELECT * FROM friend WHERE user = %s AND friend = %s"
#         cursor.execute(query, (user, friend))
#         cursor.fetchall()
#         if cursor.rowcount > 0:
#             return jsonify({"message": f"{user} and {friend} are already friends", "status": 409}), 409
#         else:
#             query = "INSERT INTO friend (user,friend) VALUES (%s, %s)"
#             cursor.execute(query, (user, friend))
#             cursor.execute(query, (friend, user))  # Add this line to add the reverse friendship
#             cnx.commit()
#             cursor.close()
#             cnx.close()
#             graph.add_edge(user, friend)
#             graph.add_edge(friend, user)  # Add this line to add the reverse friendship
#             return jsonify({"message": f"{user} and {friend} are now friends"})
#     except mysql.connector.Error as err:
#         print(f"Connection failed: {err}")
#         return jsonify({"message": "Error connecting to the database", "status": 500}), 500
#
#
# @app.route("/get_friends/", methods=["GET"])
# def get_friends():
#     user = request.args.get("user")
#     cnx = None
#     cursor = None
#     try:
#         cnx = mysql.connector.connect(user='root', password='admin', host='localhost', database='mydb')
#         cursor = cnx.cursor()
#         query = "SELECT friend FROM friend WHERE user = %s"
#         cursor.execute(query, (user,))
#         friends = cursor.fetchall()
#         friends = [friend[0] for friend in friends]
#         friends_details = []
#         for friend in friends:
#             query = "SELECT name, email FROM user WHERE username = %s"
#             cursor.execute(query, (friend,))
#             friend_details = cursor.fetchone()
#             friends_details.append({"username": friend, "name": friend_details[0], "email": friend_details[1]})
#         return jsonify({"friends": friends_details})
#     except mysql.connector.Error as err:
#         print(f"Connection failed: {err}")
#         return jsonify({"message": "Error connecting to the database", "status": 500}), 500
#     finally:
#         if cursor:
#             cursor.close()
#         if cnx:
#             cnx.close()
#
#
# @app.route("/display_friends_network", methods=["GET"])
# def display_friends_network():
#     username = request.args.get("username")
#     try:
#         cnx = mysql.connector.connect(user='root', password='admin', host='localhost', database='mydb')
#         cursor = cnx.cursor()
#         query = "SELECT user, friend FROM friend WHERE user =  %s or friend = %s"
#         cursor.execute(query, (username, username,))
#         friendships = cursor.fetchall()
#
#         # Get the names of the users
#         query = "SELECT username, name FROM user"
#         cursor.execute(query)
#         user_names = cursor.fetchall()
#         user_names = {username: name for username, name in user_names}
#         listOfNames = {}
#
#         for friendship in friendships:
#             listOfNames[friendship[0]] = user_names[friendship[0]]
#
#         G = nx.Graph()
#         G.add_edges_from(friendships)
#         plt.figure(figsize=(10, 10))
#         # Use the user_names variable as the labels
#         nx.draw(G, with_labels=True, labels=listOfNames)
#         plt.show()
#
#         cursor.close()
#         cnx.close()
#     except mysql.connector.Error as err:
#         print(f"Connection failed: {err}")
#         return jsonify({"message": "Error connecting to the database", "status": 500}), 500
#
#
# @app.route("/update_user", methods=["PUT"])
# def update_user():
#     data = request.get_json()
#     username = data["username"]
#     name = data["name"]
#     email = data["email"]
#     dp_url = data["dp_url"]
#     try:
#         cnx = mysql.connector.connect(user='root', password='admin', host='localhost', database='mydb')
#         cursor = cnx.cursor()
#         query = "UPDATE user SET name = %s, email, dp_url,  = %s WHERE username = %s"
#         cursor.execute(query, (name, email, dp_url, username))
#         cnx.commit()
#         cursor.close()
#         cnx.close()
#         graph.nodes[username]['name'] = name
#         graph.nodes[username]['email'] = email
#         return jsonify({"message": f"User {username} updated successfully", "status": 200}), 200
#     except mysql.connector.Error as err:
#         print(f"Connection failed: {err}")
#         return jsonify({"message": "Error connecting to the database", "status": 500}), 500
#
#
# @app.route('/is_friends', methods=['POST'])
# def is_friends():
#     username = request.json['username']
#     friend_username = request.json['friendUsername']
#
#     try:
#         connection = connect(
#             host='localhost',
#             user='root',
#             password='admin',
#             database='mydb'
#         )
#
#         cursor = connection.cursor()
#
#         query = f"SELECT * FROM friend WHERE (user='{username}' AND friend='{friend_username}') OR (user='{friend_username}' AND friend='{username}')"
#         cursor.execute(query)
#         result = cursor.fetchone()
#
#         if result:
#             cursor.close()
#             connection.close()
#             return jsonify({'status': 200, 'message': 'They are friends'})
#         else:
#             cursor.close()
#             connection.close()
#             return jsonify({'status': 404, 'message': 'They are not friends'})
#
#     except Error as e:
#         return jsonify({'status': 500, 'message': str(e)})
#
#
# @app.route("/delete_user", methods=["DELETE"])
# def delete_user():
#     data = request.get_json()
#     username = data["username"]
#     try:
#         cnx = mysql.connector.connect(user='root', password='admin', host='localhost', database='mydb')
#         cursor = cnx.cursor()
#
#         # Delete Friendship
#         query = "DELETE FROM friend WHERE user = %s OR friend = %s"
#         cursor.execute(query, (username, username))
#         cnx.commit()
#
#         # Delete User
#         query = "DELETE FROM user WHERE username = %s"
#         cursor.execute(query, (username,))
#         cnx.commit()
#
#         # graph.remove_node(username)
#         cursor.close()
#         cnx.close()
#         return jsonify({"message": f"User {username} and its friendships have been deleted"})
#     except mysql.connector.Error as err:
#         print(f"Connection failed: {err}")
#         return jsonify({"message": "Error connecting to the database", "status": 500}), 500
#
#
# @app.route("/get_user", methods=["GET"])
# def get_user():
#     username = request.args.get("username")
#     try:
#         cnx1 = mysql.connector.connect(user='root', password='admin', host='localhost', database='mydb')
#         cursor1 = cnx1.cursor()
#         query = "SELECT name, email, username,dp_url FROM user WHERE username = %s"
#         cursor1.execute(query, (username,))
#         user_details = cursor1.fetchone()
#         # Fetch friends of the user
#         friends_query = "SELECT friend FROM friend WHERE user = %s"
#         cursor1.execute(friends_query, (username,))
#         friends = cursor1.fetchall()
#         # Fetch friend details
#         friend_details = []
#         for friend in friends:
#             friend_query = "SELECT name, email, username, dp_url FROM user WHERE username = %s"
#             cursor1.execute(friend_query, (friend[0],))
#             friend_detail = cursor1.fetchone()
#             friend_details.append({"name": friend_detail[0], "username": friend_detail[2], "imgUrl": friend_detail[3]})
#         cursor1.close()
#         cnx1.close()
#         if user_details is None:
#             return jsonify({"message": f"User {username} not found", "status": 404}), 404
#         else:
#             return jsonify({"name": user_details[0], "email": user_details[1], "username": user_details[2],
#                             "imgUrl": user_details[3], "friends": friend_details})
#     except mysql.connector.Error as err:
#         print(f"Connection failed: {err}")
#         return jsonify({"message": "Error connecting to the database", "status": 500}), 500
#
#
# def get_users_from_db():
#     # Connect to the MySQL database
#     cnx = mysql.connector.connect(user='your_username', password='your_password', host='your_host',
#                                   database='your_database')
#     cursor = cnx.cursor()
#
#     # Select all users from the 'users' table
#     query = "SELECT * FROM users"
#     cursor.execute(query)
#     users = cursor.fetchall()
#
#     # Put the users into a graph
#     for user in users:
#         graph[user[0]] = {"name": user[1], "email": user[2]}
#
#     # Close the cursor and connection
#     cursor.close()
#     cnx.close()

if __name__ == '__main__':
    # get_users_from_db()
    app.run(host='0.0.0.0', port=5000, debug=True)

import math
import sys

def calculate_prediction_point(actual, previous):
    deltaX = actual[0] - previous[0]
    deltaY = actual[1] - previous[1]

    prediction_x = actual[0] + deltaX
    prediction_y = actual[1] + deltaY

    return(prediction_x, prediction_y)



def calculate_distance(actual, previous):
    return math.sqrt(math.pow(actual[0] - previous[0], 2) + math.pow(actual[1] - previous[1], 2))



def calculate_distances_matrix(detected_people, delayed_people):
    matrix = []
    for oldp in detected_people:
        for newp in delayed_people:
            matrix.append(calculate_distance(oldp.predictionPoint, newp.centroid))
    
    return matrix


def get_min(list):
    min_value = sys.float_info.max
    index = -1
    i = -1
    for element in list:
        i = i+1
        if element < min_value:
            min_value = element
            index = i

    return (min_value, index)

def associate_points(detected_people, delayed_people, img_width, img_height):

    distances = calculate_distances_matrix(detected_people, delayed_people)

    tracked_people = []

    min_dist, i = get_min(distances)
    

    while len(detected_people)>0 and len(delayed_people) and min_dist<img_width/8 :

        #indice dell' valore minimo nell' array delle distanze
        min_dist, min_index = get_min(distances)

        #numero di colonne della matrice (=numero di elementi in delayed_people)
        columns_matrix = int(len(delayed_people))

        
        column_index = int(min_index % columns_matrix)
        row_index = int((min_index - column_index)/columns_matrix)

        detected_person = detected_people[row_index]
        delayed_person = delayed_people[column_index]

        #associo le persone che si trovano alla distanza minima
        prediction_point = calculate_prediction_point(delayed_person.centroid, detected_person.centroid)

        delayed_person.previousPoint = detected_person.centroid
        delayed_person.predictionPoint = prediction_point
        delayed_person.crossed = (delayed_person.position != detected_person.position)

        tracked_people.append(delayed_person)


    

        #reduce matrix

        #primo elemento della riga che devo eliminare
        row_begin = min_index - column_index
        #ultimo elemento della riga che devo eliminare
        row_end = row_begin + columns_matrix

        #elimino la riga dalla matrice delle distanze
        del distances[row_begin:row_end]

        #elimino i punti associati dai rispettivi array
        detected_people.pop(row_index)
        delayed_people.pop(column_index)

        #elimino la colonna dalla matrice delle distanze
        n = int(len(delayed_people))

        for i in range(len(detected_people)):
            distances.pop(column_index)
            column_index = column_index + n


    return tracked_people
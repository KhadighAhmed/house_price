def preproces(type, fur, region, city):
    if fur == 'Yes':
        fur = 1
    if fur == 'No':
        fur = 0
    if city == 'Alexandria':
        city = 0
    if city == 'Cairo':
        city = 1
    if city == 'Giza':
        city = 2
    if type == 'Apartment':
        type = 0
    if type == 'Chalet':
        type = 1
    if type == 'Duplex':
        type = 2
    if type == 'Penthouse':
        type = 3
    if type == 'Room':
        type = 4
    if type == 'Stand Alone Villa':
        type = 5
    if type == 'Studio':
        type = 7
    if type =='Town House':
        type=8
    if type =='Twin House':
        type=9
    if region == '15 May City':
        region = 0
    if region == '6th of October':
        region = 1
    if region == 'Abasiya':
        region = 2
    if region == 'Abu Qir':
        region = 3
    if region == 'Abu Talat':
        region = 4
    if region == 'Agouza':
        region = 5
    if region == 'Al Hadrah':
        region = 6
    if region == 'Al Ibrahimiyyah':
        region = 7
    if region == 'Al Manial':
        region = 8
    if region == 'Ard El Lewa':
        region = 9
    if region == 'Asafra':
        region = 10
    if region == 'Awayed':
        region = 11
    if region == 'Azarita':
        region = 12
    if region == 'Badr City':
        region = 13
    if region == 'Bahray - Anfoshy':
        region = 14
    if region == 'Basateen':
        region = 15
    if region == 'Bolkly':
        region = 16
    if region == 'Borg al-Arab':
        region = 17
    if region == 'Boulaq Dakrour':
        region = 18
    if region == 'Camp Caesar':
        region = 19
    if region == 'Cleopatra':
        region = 20
    if region == 'Dar al-Salaam':
        region = 21
    if region == 'Dokki':
        region = 22
    if region == 'Downtown Cairo':
        region = 23
    if region == 'Faisal':
        region = 24
    if region == 'Fleming':
        region = 25
    if region == 'Garden City':
        region = 26
    if region == 'Gesr Al Suez':
        region = 27
    if region == 'Gianaclis':
        region = 28
    if region == 'Giza District':
        region = 29
    if region == 'Glim':
        region = 30
    if region == 'Hadayek 6th of October':
        region = 31
    if region == 'Hadayek al-Ahram':
        region = 32
    if region == 'Hadayek al-Kobba':
        region = 33
    if region == 'Haram':
        region = 34
    if region == 'Heliopolis':
        region = 35
    if region == 'Helmeyat El Zaytoun':
        region = 36
    if region == 'Helwan':
        region = 37
    if region == 'Imbaba':
        region = 38
    if region == 'Kabbary':
        region = 39
    if region == 'Kafr Abdo':
        region = 40
    if region == 'Katameya':
        region = 41
    if region == 'Maamoura':
        region = 42
    if region == 'Madinaty':
        region = 43
    if region == 'Mandara':
        region = 44
    if region == 'Manshiyya':
        region = 45
    if region == 'Matareya':
        region = 46
    if region == 'Miami':
        region = 47
    if region == 'Mohandessin':
        region = 48
    if region == 'Moharam Bik':
        region = 49
    if region == 'Moneeb':
        region = 50
    if region == 'Montazah':
        region = 51
    if region == 'Mostakbal City':
        region = 52
    if region == 'Nakheel':
        region = 53
    if region == 'Nasr City':
        region = 54
    if region == 'New Cairo - El Tagamoa':
        region = 55
    if region == 'New Capital City':
        region = 56
    if region == 'New Nozha':
        region = 57
    if region == 'Obour City':
        region = 58
    if region == 'Qasr al-Nil':
        region = 59
    if region == 'Raml Station':
        region = 60
    if region == 'Ras El Tin':
        region = 61
    if region == 'Rehab City':
        region = 62
    if region == 'Saba Pasha':
        region = 63
    if region == 'Salam City':
        region = 64
    if region == 'San Stefano':
        region = 65
    if region == 'Schutz':
        region = 66
    if region == 'Seyouf':
        region = 67
    if region == 'Sheikh Zayed':
        region = 68
    if region == 'Sheraton':
        region = 69
    if region == 'Shorouk City':
        region = 70
    if region == 'Shubra':
        region = 71
    if region == 'Sidi Beshr':
        region = 72
    if region == 'Sidi Gaber':
        region = 73
    if region == 'Sporting':
        region = 74
    if region == 'West Somid':
        region = 75
    if region == 'Zahraa Al Maadi':
        region = 76
    if region == 'Zamalek':
        region = 77
    if region == 'Zezenia':
        region = 78
    return type, fur, region, city

# import joblib
# import numpy as np
# model = joblib.load('final_model.pkl')
#
#
# h_type = 'Apartment'
# h_size = 200
# h_bedrooms = 2
# h_bathrooms = 2
# h_floor = 5
# h_fur = "Yes"
# h_region = 'Zamalek'
# h_city = 'Cairo'
#
# no_of_rooms = h_bathrooms+h_bedrooms
#
# h_type, h_fur, h_region, h_city = preproces(h_type, h_fur, h_region, h_city)
# output = np.array([h_type, h_size, h_bedrooms, h_bathrooms, h_floor, h_fur, h_region, h_city, no_of_rooms])
# price = model.predict(output.reshape(1, -1))
#
# print(price[0])
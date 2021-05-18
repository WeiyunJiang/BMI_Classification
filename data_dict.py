data_dict = {}
with open("data_with_feature.csv") as f:
    
    lis = [line.split('[') for line in f]        # create a list of lists
   
    for element in lis[1:]:
        key = element[0].replace(',', '').split(',')[0]
        
        value = element[1].replace(']', '')
        value = value.replace('\'', '').split(',')
        #print(value)
        
        values = []
        for x in value:
            x = x.replace(' ', '')
            if x[-1] == "n":
                x = x[:-2]
            values.append(x)
        
        data_dict[key] = values
    
    for key, value in data_dict.items():
        value[-1] = value[-1][:-2]
        
        value[1:] = [float(x) for x in value[1:]]
        data_dict[key] = value
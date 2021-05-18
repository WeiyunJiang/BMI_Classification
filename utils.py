import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dict_csv(csv_name):
    data_dict = {}
    with open(csv_name) as f:
        
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
        
    return data_dict
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 00:45:15 2020

Tabu Search with Least Cost Algorithm for U-SA-p-HLP (Case study of 76 cities in Borneo)


@author: Edric Valentino
"""
import sys
import random
import numpy as np
from collections import defaultdict
from itertools import islice
import xlrd
from geopy.distance import geodesic

           
'''             SETTING PARAMETER         '''
# jumlah node
n = 76
# himpunan node 
N = [i for i in range (n)] 
# diskon transportasi interhub
hubTransDisc = 0.5 
# jumlah hub yang akan dibangun
p = 8
# jumlah iterasi maksimal untuk restart diversifikasi
maxRun = p 
# jumlah iterasi maksimal untuk fase intensifikasi
maxCount = 150
#tabu length
tabuTenure=20

            #FUNGSI RANDOM#
rnd = np.random # fungsi random
rnd.seed(463) # biar ga berubah

            #INPUT EXCEL FILE#          
borneo = xlrd.open_workbook('C:/Users/Lenovo/iCloudDrive/TA HLP/Data Edric.xlsx')


            #LOKASI KOTA (LATITUDE, LONGITUDE)#
latlong = borneo.sheet_by_name('latlong')
lat = [float(i) for i in latlong.col_values(2,1)]
long = [float(i) for i in latlong.col_values(3,1)]
print(lat[0])
print(long[0])
dictCoordNode ={j: (lat[j], long[j]) for j in N}

            #MATRIKS JARAK (KM)#
dictDistanceMatrix = {(i,j): 0 for i in N for j in N}
for i in N:
    for j in N:
        dictDistanceMatrix[(i,j)] = geodesic(dictCoordNode[i], dictCoordNode[j]).km
        
            #BIAYA PEMBANGUNAN HUB DI SETIAP KOTA (RP)#
setupcost = borneo.sheet_by_name('setup cost')
hubsetup = [i for i in setupcost.col_values(2,2)]
dictHubSetupCost = {i: hubsetup[i] for i in N}

            #MATRIKS TRANSFER MUATAN ANTAR KOTA (KG)#
loadtransfer = borneo.sheet_by_name('load transfer')
dictLoadTransfer = {(i,j):0 for i in N for j in N}
b = [i for i in loadtransfer.col_values(0,2)]
a = [j for j in loadtransfer.row_values(0,2)]
for i, k in enumerate(b):
    for j, l in enumerate(a):
        dictLoadTransfer[l,k] = int(loadtransfer.cell(i+2,j+2).value)

            #MATRIKS ONGKOS PENGIRIMAN ANTAR KOTA (RP/KM)#
costtransfer = borneo.sheet_by_name('transfer cost')
dictCostTransfer = {(i,j):0 for i in N for j in N}
b = [i for i in costtransfer.col_values(0,2)]
a = [j for j in costtransfer.row_values(0,2)]
for i, k in enumerate(b):
    for j, l in enumerate(a):
        dictCostTransfer[l,k] = int(costtransfer.cell(i+2,j+2).value)



'''                   P R O C E D U R E                 '''
def generateStartingSolution(j, nodes, n, hub):
    random.seed(j)
    selectedHub = random.sample(nodes, hub) # generate random hub
    serviceNodes = [v for v in nodes if v not in selectedHub] # pisahkan service nodes dari hub
    rnd.seed(j+1) 
    rndServiceNodes = random.shuffle(serviceNodes) # mengacak service node untuk di kelompokkan 
    assignedNodes = [serviceNodes[x:x+7] for x in range(0, len(serviceNodes), 7)] ## mengelompokkan service nodes ke 10 grup sama bagian
    for i in range (hub):
        assignedNodes[i].insert(0, selectedHub[i])
    currentHubAssignment = dict(zip(selectedHub, assignedNodes)) # nugasin hub ke service nodes
    s = currentHubAssignment 
    return s

def calcTotalCost(nodes, solution, distancex, load, cost, setup, disc):
    #hitung ongkos terima dan ongkos kirim dari service node ke hub, pokoknya segala traffic yang ada di arc antara hub dan service node
    ongkosTerimaKirim = 0
    for i in solution:
        for j in range(len(solution[i])):
            for k in nodes:
                ongkosTerimaKirim += cost[i, solution[i][j]] * distancex[i, solution[i][j]] * (load[solution[i][j], k] + load[k, solution[i][j]])
                a = ongkosTerimaKirim
    #hitung ongkos pembangunan hub
    ongkosPembangunanHub = 0
    for i in solution:
        ongkosPembangunanHub += setup[i]
        b = ongkosPembangunanHub
    # hitung ongkos traffic interhub
    ongkirInterHub = 0
    for k in solution:
         for l in solution:
             for i in solution[k]:
                 for j in solution[l]:
                     ongkirInterHub += load[i, j] * distancex[k,l] * cost[k,l] * disc
                     c = ongkirInterHub
    d = a + b + c # menghitung total cost service arc, interhub arc, biaya pembangunan
    return d

def hubNeighborhood(solution):
    Neighbor = [(i, j) for i in solution for j in N if j not in solution]
    counterForNeighbor = 0
    allNeighborForI = np.empty((0,len(solution)))
    keys = solution.keys()
    for i in Neighbor:
        X_Swap = []
        A_Counter = Neighbor[counterForNeighbor] 
        A_1 = A_Counter[0] 
        A_2 = A_Counter[1] 
        for j in keys:
            if j == A_1:
                X_Swap = np.append(X_Swap,A_2)
            elif j == A_2:
                X_Swap = np.append(X_Swap,A_1)
            else:
                X_Swap = np.append(X_Swap,j)
            X_Swap = X_Swap[np.newaxis] # New "X0" after swap
        allNeighborForI = np.vstack((allNeighborForI,X_Swap)) # Stack all the combinations
        counterForNeighbor = counterForNeighbor+1
    a = allNeighborForI.tolist()
    b = [[int(float(n))for n in lst]for lst in a]
    return b

def chunks(data, SIZE=2000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def selectServiceNode(solution, distancex, nodes, cost, load):
    abc = {}
    for i in nodes:
        for k in range(len(solution)):
            if load[i, solution[k]] == 0:
                abc[(i, solution[k])] = distancex[i, solution[k]] * cost[i, solution[k]] * sys.maxsize
            else:
                abc[(i, solution[k])] = distancex[i, solution[k]] * cost[i, solution[k]] * load[i, solution[k]]
    aa = []
    for g in chunks(abc, len(solution)):
        a = min(g.keys(), key=(lambda k: g[k]))
        aa.append(a)
    allocate = defaultdict(dict)
    for i in range(len(solution)):
        nearest = []
        for j in range(len(aa)):
            if aa[j][1] == solution[i]:
                bb = aa[j][0]
                nearest.append(bb)
        allocate[solution[i]] = nearest
    return allocate

def solutionNeighborhood(nodes, distancex, neighbor, load, cost, setup, disc):
    a = [neighbor[i] for i in range (len(neighbor))]#misahin list hub dari nested list hubBeighbor
    e = []
    abcd = defaultdict(dict)
    for i in range(len(a)):
        c = selectServiceNode((a[i]), distancex, nodes, cost, load)
        d = calcTotalCost(nodes, c, distancex, load, cost, setup, disc)
        e.append(d)
        f = c
        f.update({'OFV' : d})
        abcd[i]= f
    return abcd

def dictToArray(solution):
    #abc = solution['OFV']
    a = solution.popitem()
    b = list(solution.keys())
    c = np.array(b)
    #d = np.append(c, abc)
    e = c.reshape(1,len(b))
    return e

def in_nested_list(my_list, item):
    if item in my_list:
        return True
    else:
        return False

#          M A I N  P R O G R A M      #
if __name__ == '__main__' : 
    divert = 0
    globalSolutionCollect = defaultdict(dict)
    globalCostCollect = dict()
    minSolC = defaultdict(dict)
    minCostC = dict()
    globalSolList = dict()
    globalNeighborCollect = dict()
    globalCostNeighbor = dict()
    for j in range(maxRun+1):
        print()
        print("--> This is the %i" % divert, "th Diversification <--")
        currentSolution = generateStartingSolution(j, N, n, p)
        currentTotalCost = calcTotalCost(N, currentSolution, dictDistanceMatrix, dictLoadTransfer, dictCostTransfer, dictHubSetupCost, hubTransDisc)
        iteration = 2
        solutionCollect = defaultdict(dict)
        costCollect = dict()
        neighborCost = defaultdict(dict)
        neighborCollect = dict()
        neighborCostCollect = dict()
        solutionLists = np.empty((0,len(currentSolution)))
        Tabu_Items = []
        Taboo_Items = []
        brow = [[el] for el in currentSolution.keys()]
        print(brow)
        print(Tabu_Items)
        solutionCollect[0] = currentSolution
        costCollect[0] = currentTotalCost
        currentSolution.update({'OFV': currentTotalCost})
        awal = dictToArray(currentSolution)
        print("--> Initial Solution <--")
        print(currentSolution)
        print(currentTotalCost)
        print()
        print("--> This is the 1 st Iteration <--")
        hubNeighbor = hubNeighborhood(currentSolution)
        solutionNeighbor = solutionNeighborhood(N, dictDistanceMatrix, hubNeighbor, dictLoadTransfer, dictCostTransfer, dictHubSetupCost, hubTransDisc)
        neighborCollect[1]= solutionNeighbor
        for k in range(len(hubNeighbor)):
            neighborCostCollect[k] = solutionNeighbor[k]['OFV']
        neighborCost[1] = neighborCostCollect
        sortedBestNeighbor = sorted(solutionNeighbor, key=lambda x: (solutionNeighbor[x]['OFV']))
        t = 0
        best = sortedBestNeighbor[t]
        sol = solutionNeighbor[best]
        abc = sol['OFV']
        e = dictToArray(sol)
        solutionLists = np.vstack((e, solutionLists))
        solutionCollect[1] = sol
        costCollect[1] = abc
        currentSolution = sol
        keluar = list(np.setdiff1d(awal,e))
        print('keluar = ',keluar)
        Taboo_Items.append(keluar)
        masuk = list(np.setdiff1d(e,awal))
        print('masuk = ',masuk)
        Tabu_Items.append(masuk)
        awal = e
        print(sol)
        print(abc)
        print(awal)
        for i in range(2, maxCount):
            print()
            print("--> This is the %i" % iteration, "th Iteration <--")
            hubNeighbor = hubNeighborhood(currentSolution)
            solutionNeighbor = solutionNeighborhood(N, dictDistanceMatrix, hubNeighbor, dictLoadTransfer, dictCostTransfer, dictHubSetupCost, hubTransDisc)
            neighborCollect[i] = solutionNeighbor
            for k in range(len(hubNeighbor)):
                neighborCostCollect[k] = solutionNeighbor[k]['OFV']
            neighborCost[i] = neighborCostCollect
            sortedBestNeighbor = sorted(solutionNeighbor, key=lambda x: (solutionNeighbor[x]['OFV']))
            t = 0
            best = sortedBestNeighbor[t]
            sol = solutionNeighbor[best]
            og_sol = abc
            abc = sol['OFV']
            og_awal = e
            e = dictToArray(sol)
            masuk = list(np.setdiff1d(e, og_awal))
            print('masuk = ',masuk)
            keluar = list(np.setdiff1d(og_awal, e))
            print('keluar = ',keluar)
            awal = e
            check = in_nested_list(Taboo_Items, masuk)
            while check is True: # If current solution is in Tabu list
                if abc < og_sol:
                    break
                else:
                    print(og_awal)
                    t = t+1
                    best = sortedBestNeighbor[t]
                    sol = solutionNeighbor[best]
                    abc = sol['OFV']
                    e = dictToArray(sol)
                    print(e)
                    masuk = list(np.setdiff1d(e,og_awal))
                    keluar = list(np.setdiff1d(og_awal,e))
                    print('keluar = ',keluar)
                    print('masuk = ',masuk)
                    check = in_nested_list(Taboo_Items, masuk)
                    print(Taboo_Items)
            if keluar in Taboo_Items:
                kk = Taboo_Items.index(keluar)
                del Taboo_Items[kk]
            if len(Taboo_Items) >= tabuTenure:
                del Taboo_Items[0]
            Tabu_Items.append(masuk)
            Taboo_Items.append(keluar)
            print(Taboo_Items)
            solutionLists = np.vstack((e, solutionLists))
            solutionCollect[i] = sol
            costCollect[i] = abc
            print(sol)
            print(abc)
            print(e)
            iteration = iteration+1
            currentSolution = sol
        Z = min(costCollect, key=costCollect.get)
        finalSolution = solutionCollect[Z]
        finalCost = costCollect[Z]
        print()
        print('--> This is the optimum solution found <--')
        print('--> Format (hub: service node) <--')
        print('Selected hub: ', finalSolution.keys()) 
        print('Configuration: ', finalSolution)
        print('Cost: ', finalCost)
        globalNeighborCollect[j] = neighborCollect
        globalCostNeighbor[j] = neighborCost
        globalSolList[j] = solutionLists
        globalSolutionCollect[j] = solutionCollect
        globalCostCollect[j] = costCollect
        minSolC[j] = finalSolution
        minCostC[j] = finalCost
        divert += 1
    G = min(globalCostCollect, key=globalCostCollect.get)
    FINAL = globalCostCollect[G]
    print('FINAL : ', FINAL)
        
        
        

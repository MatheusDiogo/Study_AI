import numpy  # importa a biblioteca numpy, que é muito usada em python para fazer cálculos
from random import randrange, choice # importa algumas funções da biblioteca random que serão usadas  pelo algorítmo
import pandas as pd # importa biblioteca usada para comunicação com alguma interface, por meio de um arquivo csv


# Hiperparâmetros de comunicação com alguma interface
nova_populacao_aleatoria = 0 # coloque '1' se a nova população for dada aleatoriamente e '0' se for usar o vetor dado a baixo
                             # sempre usar um vetor com uma quantidade de elementos que seja mútiplo de "max_num_elevador"

populacao_entrada = [5, 8, 2, 8, 5, 10, 10, 2, 1, 5, 8, 4, 4, 2, 10, 2, 3, 4, 1, 2, 1, 2, 2, 2] #


# Hiperparâmetros do Problema
andar_inicial = 0# andar inicial do elevador, se for o térreo, coloque 0
num_andares = 30# número de andares do prédio
num_barris = 24# número de barris que vão pegar o elevador nesse instante
max_num_elevador = 8# número máximo de barris que 1 elevador suporta
num_elevadores = 3# número de elevadores no prédio
d_andar = 3# distância entre andares, em metros
v_max = 2.5# velocidade máxima em m/s que o elevador pode chegar
acel = 0.5# aceleração do elevador em m/s²
tempo_parado = 1# tempo médio que o elevador demora parado em um andar, em segundos


# Hiperparâmetros do Algoritmo Genético
sol_per_pop = 8 # tamanho da população inicial
num_parents_mating = 4 # número de pais que serão considerados os melhores e liderarão a busca pela nova população
cross_rate = 50 # probabilidade de haver um crossover, sua unidade é em %
mutation_rate = 10 # probabilidade de haver uma mutação, sua unidade é em %
tamanho_populacao = 20 # número de possibilidades que existirão em cada ciclo
num_generations = 10 # número de ciclos


#Parâmetros do programa 
#esses ajudam o algoritmo a funcionar mas o operador não precisa se preocupar
tamanho_pop_sorteio = 2 # sorteia sempre de dois em dois
numpy.random.seed(seed=3)  #definir para o usuário
d_max = (v_max*v_max)/(2*acel) # deslocamento max durante o tempo de acleração, é dado em função da velocidade máxima e usado no cálculo da função custo
                              # fórmula derivada da Eq de Torricelle
    
# Python code to count the number of occurrences 
def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count 


def elimina_igual(lst):
    # from list  
    res = [] 
    [res.append(x) for x in lst if x not in res] 
    return res

def completa_faltantes(parent, result):  #completa com os valores que estavam repetidos e não entraram 
                                                     #parent: vetor com a linha de referência, result: vetor que tinha sido gerado incompleto
    valores_unicos = elimina_igual(parent)
    for valor in range(len(valores_unicos)):
        a = countX(result, valores_unicos[valor])
        b = countX(parent, valores_unicos[valor])
        c = b-a
        if c > 0:
            for i in range(c):
                result = numpy.append(result, valores_unicos[valor])  
    return result

def crossover(parent):
    result_chromossome = []
    tam_pop = len(parent[0])
    chromossome_size = (1, tam_pop)
    result_chromossome = numpy.random.uniform(low=0.0, high=0.0, size=chromossome_size)
    crossover_point = randrange(tam_pop)
    first_piece = list(parent[0][:crossover_point])
    result_chromossome = first_piece+[x for x in parent[1] if x not in first_piece]
    result_chromossome = numpy.asarray(result_chromossome)
    result_chromossome = completa_faltantes(parent[0], result_chromossome)
    return result_chromossome

def mutate(chromossome):
    chromossome = chromossome.astype(int) #transforma de array de float para inteiros
    lista = chromossome.tolist() # transforma em lista para usar a função "sample"
    i, j = numpy.random.randint(low=0, high=len(chromossome), size=2) # escolhe duas amostras da lista (cromossomo)
    lista[i], lista[j] = lista[j], lista[i]
    lista_saida = numpy.asarray(lista)
    return lista_saida


def create_new_population(populacao, tamanho_pop_sorteio, tamanho_populacao, c_rate, mutation_rate):
    new_population_a = []
    add_to_population = new_population_a.append
    #population_len = len(populacao)
    while len(new_population_a) < tamanho_populacao:
        parents = sorteio(tamanho_pop_sorteio, populacao)
        if randrange(101) <= c_rate:
            new_chromossome = crossover(parents)
        else:
            new_chromossome = choice(parents)
        if randrange(101) <= mutation_rate:
            new_chromossome = mutate(new_chromossome)
        add_to_population(new_chromossome)

    return new_population_a



def sorteio(tamanho_pop_sorteio, populacao):
    competitor = []
    competitors_size = (tamanho_pop_sorteio, len(populacao[0]))
    competitor = numpy.random.uniform(low=0.0, high=0.0, size=competitors_size)
    tamanho = len(populacao)
    for k in range(tamanho_pop_sorteio):
        competitor_num = numpy.random.randint(low=0, high=tamanho, size=1)    
        #print(competitor_num)
        competitor[k] = populacao[competitor_num]
        #print(competitor)
    return competitor



def cal_pop_fitness(populacao, d_max, d_andar, acel, max_num_elevador):
    fitness = []
    fitness_viagem = []
    fitness_size = (len(populacao), len(populacao[0]))
    fitness = numpy.random.uniform(low=0.0, high=0.0, size=fitness_size)  # cria tabela de fitness do mesmo tamanho da população
    #foi usado o uniforme para garantir valores em float
    multiplos = len(populacao)/max_num_elevador
    for i in range(len(populacao)): # ver tamanho da população
        fitness_lotacao = 0
        #Inicio fitness velocidade: Calculo da função custo para o tempo usando as equações de movimento da Física clássica 
        for j in range(len(populacao[0])-1): # ver o tamanho de um caminho qualquer para colocar no 'for'
            if ((j+1)//max_num_elevador >=1) & ((j+1)%max_num_elevador == 0): #não soma as distâncias entre o final de um elevador e começo do outro, considera cada um como um ponto sem dimensões
                num_andares = 0    # aqui 
            else:    
                num_andares = abs(populacao[i,j] - populacao[i,j+1])
            d = num_andares*d_andar/2 # calcula a ditância de deslocamente entre os andares da vez
            if d < d_max: # vê se durante essa distância ele terá atigindo a velocidade max
                          # se não houver atingido a velocidade máxima então não terá deslocamento com velocidade constante
                s = d     # deslocamento com acelaração é feito durante todo o percurso
                s_const = 0 # não há deslocamento com velocidade constante
            if d >= d_max: # se a distância de deslocamento é maior do que a distância que demora para alcansar a velocidade máxima
                s = d_max # tem o deslocamento máximo com aceleração
                s_const = d - d_max # o resto do deslocamento é feito com velocodade constante
            if s == 0:
                a = 0
                b = 0
            else:
                a = 2*((2*s/acel)**(1/2)) # calcula o tempo com velocidade variante
                b = s_const/(acel*((2*s/acel)**(1/2))) # calcula o tempo com aceleração constante 
            fitness[i,j] = a + b + tempo_parado
            fitness_lotacao = fitness_lotacao + fitness[i,j]
          #Término fitness velocidade: fim do cálculo do fitness  
        fitness_viagem.append(fitness_lotacao)
   # CALCULAR SOMA DO FITNESS PARA SER A SAÍDA
    return fitness_viagem


def select_mating_pool(pop, fitness, num_parents):
# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    fitness_select = fitness
    parents_select = numpy.empty((num_parents_mating, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness_select == numpy.min(fitness_select))
        max_fitness_idx = max_fitness_idx[0][0]
        parents_select[parent_num, :] = pop[max_fitness_idx, :]
        fitness_select[max_fitness_idx] = -99999999999 
    return parents_select

def converte_array(populacao_nova):
    nova_pop = []
    pop_size = (len(populacao_nova), len(populacao_nova[0]))
    nova_pop = numpy.random.randint(low=0, high=1, size=pop_size)
    for v in range(len(populacao_nova)):
        nova_pop[v]=populacao_nova[v]
    return nova_pop


#Gera a população aleatória inicial - barris que chegaram para pegar o elevador
# o vetor new_population é criado aleatóriamente, porque os barris que chegam para pegar o alevador vão para diversos 
# andares, como se fosse aleatoriamente
new_population = numpy.random.randint(low=andar_inicial, high=num_andares, size=num_barris) # a função random
                                                                          #gera a população aleatória respeitando os parâmetros dados


quantidade_de_viagens = num_barris//max_num_elevador  #determina a quantidade de viagens necessária para levar todas os barris, é uma divisão sem resto
if num_barris%max_num_elevador > 0: # caso seja uma divisão com resto, aumenta mais uma viagem
    quantidade_de_viagens = quantidade_de_viagens+1

init_elevador = (quantidade_de_viagens, max_num_elevador) #formato em que a matriz "elevador" terá quando criada  
elevador = numpy.random.randint(low=0, high=1, size=init_elevador) #cria uma matriz que receberá a população dos elevadores    
    
acrescimo_pop = max_num_elevador - num_barris%(num_elevadores*max_num_elevador)

for i in range(acrescimo_pop): 
    new_population = numpy.append(new_population, andar_inicial)
    #preenche com o andar inicialn(andar "0", normalmente) as vagas restantes do elevador
      # assim o programa entende que não sairá do lugar para as vagas restantes

if(nova_populacao_aleatoria):
    print("Poulação inicial: ", new_population) 
else:
    print("Poulação inicial: ", populacao_entrada)
    new_population = populacao_entrada
print("")
    
populacao = []
num_weights = max_num_elevador*quantidade_de_viagens
pop_size = (sol_per_pop, max_num_elevador*quantidade_de_viagens) # formato em que a matriz "populacao" terá quando criada 
#Cria a população inicial
populacao = numpy.random.randint(low=0, high=1, size=pop_size) #cria um vetor da população do tamanho de "pop_size"
new_population_memoria = new_population # cria uma memória para facilitar a lógica a seguir
for j in range(sol_per_pop): #laço for que se repete por "sol_per_pop" vezes e determina toda a população inicial do algorítmo genético
    individuo = []
    new_population = new_population_memoria
    for numero_do_elevador in range(quantidade_de_viagens): #define quem irá em cada elevador, de forma aleatória
        # A lógica a baixo escolhe quem irá em cada elevador, levando em conta a população que chegou para pegar o elevador
        elevador[numero_do_elevador] = numpy.random.choice(new_population, max_num_elevador, replace=False)
        # A função "choice" escolhe valores únicos, desconsiderando os repetidos e eliminando eles, assim precisamos colocar de volta
         #o que estava repetido, pois são barris diferentes
        andares_elevador = elimina_igual(elevador[numero_do_elevador]) 
        populacao_int = new_population
        new_population = [x for x in new_population if x not in elevador[numero_do_elevador]]
        for andar in range(len(andares_elevador)): #para cada andar do prédio, verifica quantos barris iam para ele da população
             #incial e quantas dessas estão nesse elevador
            a = countX(elevador[numero_do_elevador], andares_elevador[andar])
            b = countX(populacao_int, andares_elevador[andar])
            c = b-a
            if c > 0: # caso haviam mais barris para um dado andar (definido nesse laço for) do que estão organizados nos elevadores
                       # coloca esses barris que sobram de volta na população
                for i in range(c):
                    new_population = numpy.append(new_population, andares_elevador[andar]) #coloca barris repetidos que não estão em nenhum elevador de volta na população         
        #COLOCAR TODOS NUMA MESMA LINHA E FAZER A FUNÇÃO CUSTO SOMAR DE 8 EM 8
        individuo = numpy.append(individuo, elevador[numero_do_elevador]) # isso é necessário para que o algoritmo tente otimizar o percurso de todos os elevadores como se fossesm um só
    populacao[j] = individuo
    


    
best_outputs = []
best_value = []
for generation in range(num_generations): #agora são feitas várias interações a fim de encontrar a melhor resposta
    print("Generation: ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(populacao, d_max, d_andar, acel, max_num_elevador)
    print("Fitness:")
    print(fitness)
    melhor_saida = numpy.min(cal_pop_fitness(populacao, d_max, d_andar, acel, max_num_elevador))
    print("Fitness da melhor saída: ", melhor_saida)
    best_outputs.append(numpy.min(cal_pop_fitness(populacao, d_max, d_andar, acel, max_num_elevador)))
    # The best result in the current iteration.
    #print("best_outputs")
    #print(best_outputs)
    #print("Best result : ", numpy.min(cal_pop_fitness(populacao, d_max, d_andar, acel, max_num_elevador)))
    #print (numpy.min(cal_pop_fitness(populacao, d_max, d_andar, acel, max_num_elevador)))
    #best_position = numpy.where(fitness == numpy.min(fitness))
    #print("Melhor distribuição: ", populacao[best_position])
    #best_value.append(populacao[best_position])
    #print("best_value")
    #print(best_value)

    melhores = select_mating_pool(populacao, fitness, num_parents_mating)
    print("Melhor distribuição: ", melhores[0].astype(int))
    populacao_nova = create_new_population(melhores, tamanho_pop_sorteio, tamanho_populacao, cross_rate, mutation_rate)
    populacao = converte_array(populacao_nova)
    print("")
    
import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()

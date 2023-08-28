from pyModbusTCP.client import ModbusClient
import time
c = ModbusClient(host="localhost", port=502) # inicia a comunicação modbus entre o python e fábricasim

# O fuso recebe comandos entre 0 e 27648, sendo que para permenecer parado, deve receber o valor intermediário 27648/2
# quanto mais perto de 0, mais rápido ele se desloca pra esquerda, quanto mais perto de 27648 mais rápido
# se deslocar para direita


# Função que transforma um número inteiro em binário de 16 bits, invert e faz swap dos bits para enviar para o PLC 
def swap_int(num_int):    
    length = len(bin(num_int)) - 2
    diferenca = 16 - length
    num = [0 for a in range(16)]
    for a in range(16):
        if (a < diferenca):
            num[a] = 0
        else:     
            num[a] = int(bin(num_int)[a-diferenca+2])
    
#    print(num)        
    num_inv = [0 for a in range(16)]
    for a in range(0,16):
        num_inv[a] = num[-a-1]
#    print(num_inv)
    for a in range(8):
        num[a] = num_inv[a+8]
        
    for a in range(8,16):
        num[a] = num_inv[a-8]
        
    return num


c.open()
return_num = swap_int(14000) # digite um número para fazer o fuso se deslocar
print(return_num)
c.write_multiple_coils(32, return_num) # movendo o fuso em um sentido
time.sleep(2.0)

bound = [0, 27648]

return_num_stot = swap_int(0)
print(return_num_stot)
c.write_multiple_coils(32, return_num_stot)
time.sleep(2.0)

print("Localização atual da ferramenta: ",c.read_input_registers(0)) # ler a localização da ferramenta

# Treinamento da IA, parte 1
# Compile para configurar o modelo, o treino se dará no próximo bloco.
# Atenção: Se você treinar a IA mais de uma vez com hiperpâmetro diferentes, ela vai ficar confusa e pode não aprender bem. 
# Se for mudar os hiperparâmetros e quiser treinar do início novamente, precisa dar "Shutdown" e 
# depois "Reconnect", na aba "Kernel"

import random
import threading
import time
from rl_fuso import DDPG
import numpy as np

# Parâmetros do Ambiente - Máquina
s_dim = 4 # quantidade de variáveis de estado que o ambiente do robo nos retorna
a_dim = 1 # quantidade de motores que precisam ser controlados para mover o robo
a_bound = [0,1] # área de trabalho da rede neural
on_goal_ext = 0 # vezes que atingiu o alvo por episódio
comp_trabalho = bound[1]  # comprimento da área de trabalho do robô
# Selecione o Método de RL
rl = DDPG(a_dim, s_dim, a_bound)  # Aqui definimos o algoritmo DDPG como algorítmo de treino
time.sleep(2.0)

#Hiperparametros de Treinamento
MAX_EPISODES = 200 # 200 número de episódios para o treinamento
MAX_EP_STEPS = 40 # 40 número máximo de passos em cada episódio
ON_TRAIN = True # permite o treinamento da rede

distance_error = 300 # 300 distancia mínima em que o robo tem que estar do centro do alvo para ser considerado sucesso
times_for_done = 5 # 5 define quantas vezes é necessário atinfir o alvo em um episódio para entender que houve sucesso
peso_erro = 5 # 3 quantas vezes multiplica o erro para dar a recompensa negativa
tempo = 0.3

# Parametros (Não mudar, apenas declaração de variáveis)
dist_ant = 0
velocidade_ant = 0

# Treinamento da IA, parte 2
# Compile para treinar o modelo.
# Atenção: Se você treinar a IA mais de uma vez com hiperpâmetros diferentes, ela vai ficar confusa e pode não aprender bem. 
# Se for mudar os hiperparâmetros e quiser treinar do início novamente, precisa dar "Shutdown" e 
# depois "Reconnect", na aba "Kernel"

def write_position(ang_input_1, on_goal, dist_ant_local, velocidade_ant_local, alvo, init):
    goal = alvo
    done = False
    
    
    # Novo deslocamento
    if(init != 0):
        ang_1 = (ang_input_1 + 1)/2 # transforma a saída da rede de 0 a -1 para um valor de 0 a 1
        comando_deslocamento = (comp_trabalho * ang_1) # transforma em um valor de 0 até 27648 que é o range máximo do comando
        print("Comando deslocamento",comando_deslocamento)
        #print("comando_deslocamento: " + str(comando_deslocamento))
        return_num = swap_int(int(comando_deslocamento)) # ajusta os bits para enviar comando para o PLC via modbus
        c.write_multiple_coils(32, return_num) # manda o fuso se deslocar
    
        time.sleep(tempo) # intervalo de tempo
    
    # ler Distâncias da Ferramenta
    actual_state = c.read_input_registers(0) # localização da ferramenta - fuso
    print("Posição da ferramenta: " + str(((actual_state[0]*2000)/27648)))
    o_a1 = goal
    
   
    dist = (goal - actual_state[0]) / comp_trabalho # fuso
    recompensa = -abs(peso_erro*dist) # fuso - função fitness              
    print("Recompensa: " + str(recompensa))  # descomente se quiser ver a recompensa em cada passos
    
    if (recompensa < -10):
        recompensa = -0.5
    # done and reward
    if ( o_a1 - distance_error < actual_state[0] < o_a1 + distance_error) : # fuso
        recompensa += 1.  # recompensa positiva sempre que encontra o alvo
        on_goal = on_goal + 1
        print('__________Encontrou o alvo!_________') 

        if on_goal >= times_for_done: #define quantas vezes é necessário atinfir o alvo em um episódio para entender que houve sucesso
            done = True
            print("________________________Manteve-se no alvo!________________________")
    else:
        on_goal = 0

    #cálculo da velocidade e da aceleração
    velocidade = (dist - dist_ant_local)/tempo
    aceleracao = (velocidade - velocidade_ant_local)/tempo
    print("Distância: " + str(dist))
    
    state_for_rl = np.concatenate(([aceleracao/10], [velocidade/10], [dist], [1. if on_goal else 0.]))
    #print(state_for_rl)
    return state_for_rl, recompensa, done, on_goal, dist, velocidade


steps = []
if ON_TRAIN:  # se quiser treinar novamente, só copiar em uma nova linha a partir daqui a mudar o numero de MAX_EPISODES para 
               # um valor maior, vc pode mudar todas as outras configurações, como diminuir o distance_error
    # start training
    for i in range(MAX_EPISODES):
        print("")
        print("_____________________ Episódio N°: " + str(i) + "_____________________" )
      
       # Define uma posição aleatória para o alvo dentro do range de trabaho
        pos_init_goal_1 = random.randint(0, bound[1])
        goal = pos_init_goal_1
        
        # Define uma posição aleatória inicial para o robô dentro do range de trabaho
        pos_init_axis_1 = random.randint(0, bound[1])
        return_num_init = swap_int(pos_init_axis_1) # ajusta os bits para enviar comando para o PLC via modbus
        time.sleep(0.5)
        c.write_multiple_coils(32, return_num_stot) # para o fuso
        time.sleep(2.0)
        
        pos_init = c.read_input_registers(0) # posição incial da ferramenta
        time.sleep(0.5)
        dist_ant = (goal - pos_init[0])/bound[1] # distância inicial do alvo
        velocidade_ant = 0
        
        s, r, done, on_goal_ext, dist_ant, velocidade_ant = write_position(0, 0, dist_ant, velocidade_ant, goal, 0) # reinicia
        
        on_goal_ext = 0 # vezes que atingiu o alvo por episódio
        ep_r = 0.
        for j in range(MAX_EP_STEPS): # em cada passo a memória é enchida com possíveis situações dada pela rede "choose_action(s)"
                                       # essas ações são os passos de um episódio
            print("___ STEP: " + str(j)+ "___")
            print("Alvo: " + str(goal*2000/bound[1]))
            a = rl.choose_action(s) # faz a rede neural escolher uma ação dado o atual estado 's', 
                                       #no início ela não está treinada e escolhe ações pouco eficazes
                
            (a1r) = a
            print("Action: ")
            print(a)
            
            s_, r, done, on_goal_ext, dist_ant, velocidade_ant = write_position(a1r, on_goal_ext, dist_ant, velocidade_ant, goal, 1) # com a ação é gerado um novo estado 's_', uma nova recompensa 'r' 
                                                        # e é avisado se o robô chegou no alvo
                
            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1: # vão acontecendo passos até que alcance o objetivo ou o número máximo de passos por episódio
                print('Ep: %i | %s | ep_r: %.1f | step: %i ' % (i, '---' if not done else 'done', ep_r, j))
                c.write_multiple_coils(32, return_num_stot) # para o fuso ao final do episódio
                print(a)
                time.sleep(2.5)
                break  #interrompe o  laço "for", assim finaliza um episódio
    rl.save()

else:
    print("Aguarda treinamento")

# Configuração para Deslocar o Fuso. Aqui nós fazemos a configuração para que você desloque nos próximos blocos
# Diminua distance_error para ficar mais acurado
# sendo assim, precisamos declarar a função "write_position()" mais uma vez

distance_error = 100 # distancia mínima em que o robo tem que estar do centro do alvo para ser considerado sucesso
times_for_done = 2 # define quantas vezes é necessário atingir o alvo em um episódio para entender que houve sucesso
peso_erro = 5 # quantas vezes multiplica o erro para dar a recompensa negativa
tempo = 0.5


def write_position(ang_input_1, on_goal, dist_ant_local, velocidade_ant_local, alvo, init):
    goal = alvo
    done = False


    # Novo deslocamento, agora para o fuso
    if(init != 0):
        ang_1 = (ang_input_1 + 1)/2 # transforma a saída da rede de 0 a -1 para um valor de 0 a 1
        comando_deslocamento = comp_trabalho * ang_1 # transforma em um valor de 0 até 27648 que é o range máximo do comando
        #print("comando_deslocamento: " + str(comando_deslocamento))
        return_num = swap_int(int(comando_deslocamento)) # ajusta os bits para enviar comando para o PLC via modbus
        c.write_multiple_coils(32, return_num) # manda o fuso se deslocar

        time.sleep(tempo) # intervalo de tempo


    # ler Distâncias da Ferramenta
    actual_state = c.read_input_registers(0) # localização da ferramenta - fuso
    print("Posição da ferramenta: " + str((actual_state[0]/27648)*2000))
    o_a1 = goal

    dist = (goal - actual_state[0]) / comp_trabalho # fuso
    recompensa = -abs(peso_erro*dist) # fuso - função fitness              
    print("Recompensa: " + str(recompensa))  # descomente se quiser ver a recompensa em cada passos

    if (recompensa < -10):
        recompensa = -0.5
    # done and reward
    
    if ( o_a1 - distance_error < actual_state[0] < o_a1 + distance_error) : # fuso
        recompensa += 1.  # recompensa positiva sempre que encontra o alvo
        on_goal = on_goal + 1
        print('__________Encontrou o alvo!_________') 
        c.write_multiple_coils(32, swap_int(13824))
        time.sleep(15.0)
        

        if on_goal >= times_for_done: #define quantas vezes é necessário atinfir o alvo em um episódio para entender que houve sucesso
            done = True
            
            print("________________________Manteve-se no alvo!________________________")
    else:
        on_goal = 0

    #cálculo da velocidade e da aceleração
    velocidade = (dist - dist_ant_local)/tempo
    aceleracao = (velocidade - velocidade_ant_local)/tempo
    #print("velocidade: " + str(velocidade))
    #print("aceleracao: " + str(aceleracao))
    print("Distância: " + str(dist))

    state_for_rl = np.concatenate(([aceleracao/10], [velocidade/10], [dist], 
                                   #[dist3_x], [dist3_y], [np.sqrt(dist3_x**2+dist3_y**2)], [recompensa],
                                   [1. if on_goal else 0.]))

    return state_for_rl, recompensa, done, on_goal, dist, velocidade

# Deslocar o fuso. Aqui você pode testar a vontade, modificando o alvo no final.
# Essa rotina faz o fuso se mover de uma posição inicial para uma final, 
# veja o final dessa linha

c.write_multiple_coils(32,swap_int(0))
time.sleep(2.0)

def move_more(alvo_fuso):
    goal = alvo_fuso*bound[1]/2000 # transforma escala da cena na escala do clp
    pos_init = c.read_input_registers(0) # posição incial da ferramenta
    dist_ant = (goal - pos_init[0])/bound[1] # distância inicial do alvo
    velocidade_ant = 0

    s, r, done, on_goal_ext, dist_ant, velocidade_ant = write_position(0, 0, dist_ant, velocidade_ant, goal, 0) # reinicia
    on_goal_ext = 0 # vezes que atingiu o alvo por episódio
    ep_r = 0.
    for j in range(MAX_EP_STEPS): # em cada passo a memória é enchida com possíveis situações dada pela rede "choose_action(s)"
                                   # essas ações são os passos de um episódio
        print("___ STEP: " + str(j)+ "___")
        print("Alvo: " + str(goal*2000/27648))
        a = rl.choose_action(s) # faz a rede neural escolher uma ação dado o atual estado 's', 
                                   #no início ela não está treinada e escolhe ações pouco eficazes
                                # 'a' é a ação, que são os ângulos dos dois braços, como mostrado na função 'env.step'
                                   #o valor de 'a' é entre 0 e 1, que simboliza 0 e + 2*pi em radianos (-360° e +360°)   
        (a1r) = a
        print("Action: ")
        print(a)

        s_, r, done, on_goal_ext, dist_ant, velocidade_ant = write_position(a1r, on_goal_ext, dist_ant, velocidade_ant, goal, 1) # com a ação é gerado um novo estado 's_', uma nova recompensa 'r' 
                                                    # e é avisado se o robô chegou no alvo
    
        rl.store_transition(s, a, r, s_)

        ep_r += r
        if rl.memory_full:
            # start to learn once has fulfilled the memory
            rl.learn()

        s = s_
        if done or j == MAX_EP_STEPS-1: # vão acontecendo passos até que alcanse o objetivo ou o número máximo de passos por episódio
            print('Ep: %i | %s | ep_r: %.1f | step: %i ' % (i, '---' if not done else 'done', ep_r, j))
            c.write_multiple_coils(32, return_num_stot) # para o fuso ao final do episódio
            print(a)
            break  #interrompe o  laço "for", assim finaliza um episódio
    rl.save()



# escolha um alvo de 0 até 2000   
alvo = 1600
#time.sleep(4)
move_more(alvo) #escolhe um alvo de 0 até 2000
time.sleep(3) # intervalo de tempo
#para mover o fuso
c.write_multiple_coils(32, swap_int(13824)) # se quiser ir para o 2000 (direita), coloque 27000

######## VALIDAÇÃO ######## 
#Execute este código e clique em "Entregar" para que a plataforma valide sua solução


c.write_multiple_coils(32,swap_int(0))
time.sleep(2.0)
DI = c.read_discrete_inputs(0,3)

while(DI[2] == False):
    DI = c.read_discrete_inputs(0,3)

validating = True
    
def move_more(alvo_fuso):
    goal = alvo_fuso*bound[1]/2000 # transforma escala da cena na escala do clp
    pos_init = c.read_input_registers(0) # posição incial da ferramenta
    dist_ant = (goal - pos_init[0])/bound[1] # distância inicial do alvo
    velocidade_ant = 0

    s, r, done, on_goal_ext, dist_ant, velocidade_ant = write_position(0, 0, dist_ant, velocidade_ant, goal, 0) # reinicia
    on_goal_ext = 0 # vezes que atingiu o alvo por episódio
    ep_r = 0.
    for j in range(MAX_EP_STEPS): # em cada passo a memória é enchida com possíveis situações dada pela rede "choose_action(s)"
                                   # essas ações são os passos de um episódio
        print("___ STEP: " + str(j)+ "___")
        print("Alvo: " + str(goal*2000/27648))
        a = rl.choose_action(s) # faz a rede neural escolher uma ação dado o atual estado 's', 
                                   #no início ela não está treinada e escolhe ações pouco eficazes
                                # 'a' é a ação, que são os ângulos dos dois braços, como mostrado na função 'env.step'
                                   #o valor de 'a' é entre 0 e 1, que simboliza 0 e + 2*pi em radianos (-360° e +360°)   
        (a1r) = a
        print("Action: ")
        print(a)

        s_, r, done, on_goal_ext, dist_ant, velocidade_ant = write_position(a1r, on_goal_ext, dist_ant, velocidade_ant, goal, 1) # com a ação é gerado um novo estado 's_', uma nova recompensa 'r' 
                                                    # e é avisado se o robô chegou no alvo
        #s_, r, done = env.step(a) # a ação gera um novo estado 's_' e uma nova recompensa 'r' 
      # r = r - j/100  # otimizador, para cada passo a mais que dá, diminue a recompensa, assim tenta fazer mais rápido 

        rl.store_transition(s, a, r, s_)

        ep_r += r
        if rl.memory_full:
            # start to learn once has fulfilled the memory
            rl.learn()

        s = s_
        if done or j == MAX_EP_STEPS-1: # vão acontecendo passos até que alcanse o objetivo ou o número máximo de passos por episódio
            print('Ep: %i | %s | ep_r: %.1f | step: %i ' % (i, '---' if not done else 'done', ep_r, j))
            c.write_multiple_coils(32, return_num_stot) # para o fuso ao final do episódio
            print(a)
            break  #interrompe o  laço "for", assim finaliza um episódio
    rl.save()



# escolhe um alvo de 0 até 2000   
alvo = 1000
#time.sleep(4)
move_more(alvo) #escolhe um alvo de 0 até 2000
time.sleep(3) # intervalo de tempo
#para mover o fuso
c.write_multiple_coils(32, swap_int(13824)) # se quiser ir para o 2000 (direita), coloque 27000

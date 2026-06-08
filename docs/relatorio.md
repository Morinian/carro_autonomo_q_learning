# Relatório – EP Carro Autônomo com Q-Learning Tabular

## Integrantes

* Paula Martins de Souza
* Eric Donato
* Matheus Lima

---

# 1. Escolha dos Hiperparâmetros

## Taxa de aprendizado (α)

Foi utilizada:

```python
alpha = 0.1
```

A taxa de aprendizado controla o quanto novas experiências alteram os valores já aprendidos da tabela Q.

Escolhemos α = 0,1 por representar um equilíbrio entre velocidade de aprendizado e estabilidade. Valores maiores produziram atualizações muito agressivas, enquanto valores menores tornaram o treinamento excessivamente lento.

---

## Fator de desconto (γ)

Foi utilizado:

```python
gamma = 0.99
```

O fator de desconto determina a importância das recompensas futuras.

Como o objetivo do problema é completar uma pista longa até alcançar a linha de chegada, foi escolhido um valor próximo de 1. Dessa forma o agente valoriza ações que geram benefícios futuros, e não apenas recompensas imediatas.

---

## Política ε-greedy

Foi utilizada uma política ε-greedy com:

```python
eps_inicial = 1.0
eps_final = 0.05
```

O valor de ε decresce linearmente durante 80% do treinamento.

No início do treinamento:

```text
ε = 1.0
```

o agente explora intensamente o ambiente.

Ao final do treinamento:

```text
ε = 0.05
```

o agente utiliza principalmente o conhecimento armazenado na tabela Q, mantendo apenas uma pequena taxa de exploração.

O cálculo utilizado foi:

```python
frac = ep / decaimento_eps_episodios

agente.eps = (
    agente.eps_inicial
    - frac * (agente.eps_inicial - agente.eps_final)
)
```

---

## Orçamento de Treino

Foi utilizado:

```text
30.000 episódios por pista
```

Como existem 16 pistas de treinamento:

```text
30.000 × 16 = 480.000 episódios totais
```

Foram realizados diversos testes com quantidades menores de episódios.

Observou-se que o aumento do treinamento melhora significativamente o desempenho nas pistas de holdout até determinado ponto. Em experimentos com treinamento ainda maior (960.000 episódios), o agente apresentou degradação de desempenho, indicando convergência para políticas excessivamente conservadoras.

---

# 2. Mecânica da Exploração

O agente utiliza a estratégia ε-greedy.

Durante cada passo do treinamento:

```python
if random.random() < self.eps:
    return random.randint(0, self.n_actions - 1)

return int(np.argmax(self.Q[estado]))
```

Com probabilidade ε:

* escolhe uma ação aleatória (exploração)

Com probabilidade 1 − ε:

* escolhe a ação de maior valor Q conhecido (exploração do conhecimento adquirido)

Nenhuma técnica adicional foi utilizada, como action masking ou exploração baseada em contagem de estados.

---

# 3. Implementação

## Modelagem do MDP

### Estados

O ambiente fornece observações compostas por:

```text
[d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
```

onde:

* d_0 = distância frontal
* d_+30 = distância à frente e à direita
* d_-30 = distância à frente e à esquerda
* d_+60 = distância lateral direita
* d_-60 = distância lateral esquerda
* v_norm = velocidade normalizada

Todos os valores estão normalizados entre 0 e 1.

---

### Discretização

Como o Q-Learning tabular exige estados discretos, foi utilizada discretização uniforme com:

```python
K = 5
```

Implementação:

```python
def discretizar(obs):
    return tuple(
        min(int(v * K), K - 1)
        for v in obs
    )
```

Cada observação contínua é convertida em uma tupla de 6 inteiros.

Exemplo:

```text
[0.35, 1.00, 0.30, 0.41, 0.18, 0.50]
↓
(1, 4, 1, 2, 0, 2)
```

Essa tupla é utilizada como chave da tabela Q.

---

### Ações

O espaço de ações possui cinco ações discretas:

| Ação | Descrição        |
| ---- | ---------------- |
| 0    | Não fazer nada   |
| 1    | Acelerar         |
| 2    | Frear            |
| 3    | Virar à esquerda |
| 4    | Virar à direita  |

---

### Tabela Q

A tabela Q foi implementada utilizando:

```python
defaultdict(
    lambda: np.zeros(n_actions)
)
```

Cada estado discretizado armazena um vetor com cinco valores Q, um para cada ação possível.

---

### Atualização do Q-Learning

Foi utilizada a atualização clássica:

```python
Q(s,a) ← Q(s,a) +
α [ r + γ max Q(s',a') − Q(s,a) ]
```

Implementação:

```python
self.Q[estado][a] += self.alpha * (
    alvo - q_atual
)
```

---

## Treinamento Round-Robin

O treinamento foi realizado utilizando as pistas 01 a 16.

A cada episódio uma pista é sorteada aleatoriamente:

```python
pista = random.choice(pistas_treino)
```

Essa estratégia evita que o agente aprenda apenas a última pista treinada e reduz o problema de esquecimento catastrófico.

---

# 4. Resultados nas Pistas Holdout

## Pista 17

```text
Tempo de chegada: 95 passos
Velocidade média: 1.53
Velocidade máxima: 2.00
Estados populados: 6220
Recompensa total: 629.50
Sucesso: SIM
```

---

## Pista 18

```text
Tempo de chegada: 500 passos
Velocidade média: 0.26
Velocidade máxima: 2.00
Estados populados: 6228
Recompensa total: 12.00
Sucesso: NÃO
```

---

# 5. Análise Crítica

O agente apresentou boa capacidade de generalização na pista 17, conseguindo completar a pista sem ter sido treinado nela.

Entretanto, a pista 18 não foi concluída. O agente conseguiu evitar colisões em grande parte do episódio, mas não encontrou uma sequência de ações capaz de alcançar a linha de chegada dentro do limite de passos.

Isso evidencia uma limitação da representação baseada apenas em sensores LIDAR locais. O agente não possui conhecimento da posição global na pista nem de sua orientação absoluta, tomando decisões apenas com base nas distâncias observadas ao seu redor.

Além disso, o Q-Learning tabular possui limitações de generalização. Como os estados são discretizados, situações semelhantes podem cair em estados diferentes da tabela, reduzindo a transferência de conhecimento entre pistas distintas.

Durante os testes também foi observado que aumentar excessivamente o treinamento (960.000 episódios) não melhorou o desempenho. Nesse cenário, o agente passou a adotar comportamentos excessivamente conservadores, reduzindo sua velocidade e deixando de concluir até mesmo pistas que anteriormente conseguia resolver.

Portanto, os resultados indicam que a combinação entre representação local via LIDAR e Q-Learning tabular é capaz de aprender comportamentos úteis de navegação, mas apresenta limitações quando confrontada com pistas significativamente diferentes das observadas durante o treinamento.

---

# 6. Conclusão

Foi implementado um agente baseado em Q-Learning tabular para controlar um carro autônomo utilizando apenas informações provenientes de sensores LIDAR simulados.

O agente foi treinado em 16 pistas utilizando treinamento round-robin e avaliado em duas pistas holdout nunca vistas durante o treinamento.

Os resultados mostraram sucesso na pista 17 e falha na pista 18, evidenciando capacidade parcial de generalização. O experimento também demonstrou a importância da discretização dos estados, da política ε-greedy e da escolha adequada do orçamento de treinamento.

Apesar das limitações observadas, o agente foi capaz de aprender comportamentos de direção e navegação utilizando exclusivamente aprendizado por reforço tabular.

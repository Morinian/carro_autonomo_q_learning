# EP Carro Autônomo — Q-Learning Tabular

## 👥 Integrantes
- Matheus Lima  
- Eric Donato  
- Paula Martins de Souza  

## 🚀 Como rodar o projeto

### 1. Instalar dependências
```bash
pip install -r requirements.txt
````

---

### 2. Treinar o agente

```bash
python solucao.py --episodios-por-pista 30000
```

---

### 3. Forçar re-treinamento

```bash
python solucao.py --episodios-por-pista 30000 --recarregar
```

---

### 4. Avaliar o modelo (pistas 17 e 18)

```bash
python solucao.py
```

---

## 📁 Saídas geradas

Após execução serão criados:

* `treinamento/qlearning.pkl`
* `q_learning_pista_17.txt`
* `q_learning_pista_18.txt`

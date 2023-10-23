# QoS IC

Estes são os arquivos da minha pesquisa que busca entender como um modelo de Machine Learning deteriora na medida em que a natureza dos dados de entrada são diferentes daqueles de treinamento, e isso acontece *o tempo todo*.

De forma geral, quando uma empresa quer estatísticas, é uma questão de escolha coletar dos próprios servidores, mas não necessariamente do cliente, que pode aceitar ou não, pode deteriorar sua conexão etc. Dessa forma faz-se necessário com dados mínimos de (alguns) usuários e vastos do servidor prever a qualidade do serviço prestado ao usuário sem mesmo consultá-lo, dessa forma maximizando sua satisfação e respeitando sua privacidade e desempenho.

Meu orientador, Pasquini, [fez uma pesquisa](https://github.com/rafaelpasquini/traces-netsoft-2017) onde mostrou ser possível prever o QoS de aplicações Streaming e Bancos de dados com métodos de Machine Learning (Regression Tree e Random Forest), dessa forma minha pesquisa busca explorar algumas premissas e diferenças na prática que ocorrem:
1. Será que precisamos de informação por segundo? Na pesquisa, meu orientador salvou as estatísticas de usuário e do cluster por segundo, se os resultados fossem esparsos e com períodos distintos, como o resultado seria deteriorado?
2. Será que precisamos de dados em momentos de pico? O tráfego de estatísticas *também* é tráfego e pode ser um gargalo da rede, pode tomar um espaço que poderia ter ajudado o desempenho de um cliente em um momento de pico... etc. Se usássemos os dados apenas de momentos ruins ou com alguma mescla, teríamos os mesmos resultados?

## Os arquivos

Note que eu uso jupyter & scripts python. São verdadeiros gigabytes de arquivos, então às vezes testo local ou no jupyter de uma forma, exploro de outra e uso uma máquina potente para replicar sem jupyter, por isso você pode notar que alguns arquivos .py parecem meio "imperativos" dada natureza do jupyter, enquanto outros (como o replication.py) tem um ar mais generalizado

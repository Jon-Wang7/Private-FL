import torch.nn as nn
import pennylane as qml
import torch

# 量子电路参数
n_qubits = 3
n_fuzzy_mem = 2
defuzz_qubits = n_qubits
defuzz_layer = 2

# 量子设备
dev1 = qml.device('default.qubit', wires=2*n_qubits-1)
dev2 = qml.device('default.qubit', wires=defuzz_qubits)

# 量子电路
@qml.qnode(dev1, interface='torch', diff_method='backprop')
def q_tnorm_node(inputs, weights=None):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.Toffoli(wires=[0,1,n_qubits])
    for i in range(n_qubits-2):
        qml.Toffoli(wires=[i+2,n_qubits+i,i+n_qubits+1])
    return qml.probs(wires=2*n_qubits-2)

@qml.qnode(dev2, interface='torch', diff_method='backprop')
def q_defuzz(inputs, weights=None):
    qml.AmplitudeEmbedding(inputs, wires=range(defuzz_qubits), normalize=True)
    for i in range(defuzz_layer):
        for j in range(defuzz_qubits-1):
            qml.CNOT(wires=[j,j+1])
        qml.CNOT(wires=[defuzz_qubits-1,0])
        for j in range(defuzz_qubits):
            qml.RX(weights[i,3*j], wires=j)
            qml.RZ(weights[i,3*j+1], wires=j)
            qml.RX(weights[i,3*j+2], wires=j)
    return [qml.expval(qml.PauliZ(j)) for j in range(defuzz_qubits)]

# 权重形状
weight_shapes = {"weights": (1, 1)}
defuzz_weight_shapes = {"weights": (defuzz_layer, 3*defuzz_qubits)}

# 权重初始化函数
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Parameter):
        torch.nn.init.normal_(m.data)


class CustomResNet(nn.Module):
    def __init__(self, name, in_channels=0, num_classes=0):
        super(CustomResNet, self).__init__()
        self.name = name
        self.num_classes = num_classes

        # Define convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(32, 64)
        self.tanh1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, 128)
        self.tanh2 = nn.ReLU()

        # Define residual blocks
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 512),
        )

        # Define final classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh2(x)

        # Residual blocks
        residual = x
        x = self.res1(x)
        x += residual
        x = self.res2(x)
        x = self.res3(x)

        # Final classification layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CNN(nn.Module):
    def __init__(self, name, in_channels=0, num_classes=0):
        super(CNN, self).__init__()
        self.name = name

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Qfnn(nn.Module):
    def __init__(self, name, in_channels=0, num_classes=0, device='cpu', config=None):
        super(Qfnn, self).__init__()
        self.name = name
        self.device = device
        
        # 从配置中获取参数
        if config is not None:
            self.n_qubits = config.get("n_qubits", 3)
            self.n_fuzzy_mem = config.get("n_fuzzy_mem", 3)
            hidden_layers = config.get("hidden_layers", [128, 64])
            dropout = config.get("dropout", 0.3)
        else:
            self.n_qubits = 3
            self.n_fuzzy_mem = 3
            hidden_layers = [128, 64]
            dropout = 0.3
        
        self.max_batch_size = 32
        
        # 1. 增强特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 2. 初始化量子设备
        self.dev1 = qml.device('default.qubit', wires=self.n_qubits)
        
        # 3. 增强量子电路
        @qml.qnode(self.dev1, interface='torch', diff_method='backprop')
        def quantum_circuit(inputs, weights=None):
            # 量子态编码
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
            
            # 添加纠缠
            for i in range(self.n_qubits-1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits-1, 0])
            
            # 参数化旋转
            if weights is not None:
                for i in range(self.n_qubits):
                    qml.RX(weights[0, i], wires=i)
                    qml.RZ(weights[1, i], wires=i)
                    qml.RX(weights[2, i], wires=i)
            
            # 添加额外纠缠
            for i in range(self.n_qubits-1):
                qml.CNOT(wires=[i, i+1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        # 4. 定义权重形状
        self.weight_shapes = {"weights": (3, self.n_qubits)}
        
        # 5. 定义网络层
        self.linear = nn.Linear(128 * 3 * 3, self.n_qubits)
        self.gn = nn.GroupNorm(1, self.n_qubits)
        self.dropout = nn.Dropout(dropout)
        
        # 6. 模糊化参数
        self.m = nn.Parameter(torch.randn(self.n_qubits, self.n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(self.n_qubits, self.n_fuzzy_mem))
        
        # 7. 计算特征数
        self.quantum_features = self.n_qubits * (self.n_fuzzy_mem ** self.n_qubits)
        
        # 8. 增强中间层
        self.hidden = nn.Sequential(
            nn.Linear(self.quantum_features, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 9. 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_layers[-1], num_classes),
            nn.Softmax(dim=1)
        )
        
        # 10. 初始化量子层
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)
        
        # 11. 初始化权重
        self.apply(weights_init)
    
    def forward(self, x):
        # 1. 特征提取
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        
        # 2. 分批处理
        batch_size = x.size(0)
        if batch_size > self.max_batch_size:
            outputs = []
            for i in range(0, batch_size, self.max_batch_size):
                end = min(i + self.max_batch_size, batch_size)
                batch_x = x[i:end]
                batch_output = self._process_batch(batch_x)
                outputs.append(batch_output)
            x = torch.cat(outputs, dim=0)
        else:
            x = self._process_batch(x)
        
        # 3. 中间层
        x = self.hidden(x)
        
        # 4. 分类
        x = self.classifier(x)
        return x
    
    def _process_batch(self, x):
        # 1. 线性变换
        x = self.linear(x)
        x = self.gn(x)
        x = self.dropout(x)
        
        # 2. 模糊化处理
        fuzzy_list = []
        for i in range(self.n_fuzzy_mem):
            diff = x.unsqueeze(2) - self.m.t()
            exp_term = torch.exp(-0.5 * (diff / self.theta.t())**2)
            fuzzy = exp_term.mean(dim=2)
            fuzzy_list.append(fuzzy)
        
        # 3. 量子计算
        q_in = torch.zeros_like(x).to(self.device)
        q_out = []
        
        for i in range(self.n_fuzzy_mem**self.n_qubits):
            loc = list(bin(i))[2:]
            if len(loc) < self.n_qubits:
                loc = [0]*(self.n_qubits-len(loc)) + loc
            for j in range(self.n_qubits):
                q_in = q_in.clone()
                q_in[:,j] = fuzzy_list[int(loc[j])][:,j]
            
            # 量子电路处理
            sq = torch.sqrt(q_in+1e-16)
            sq = torch.clamp(sq, -0.99999, 0.99999) 
            q_in = 2*torch.arcsin(sq)
            
            Q_out = self.qlayer(q_in)
            if Q_out.dim() == 1:
                Q_out = Q_out.unsqueeze(0)
            q_out.append(Q_out)
        
        # 4. 合并输出
        out = torch.cat(q_out, dim=1)
        
        # 5. 确保维度正确
        if out.shape[1] != self.quantum_features:
            out = out.view(out.shape[0], -1)
        
        return out

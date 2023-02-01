import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForPreTraining, AutoTokenizer
from normalizer import normalize
import sys
#sys.path.append(r"C:\Users\USER\AppData\Local\Programs\Python\Python311\Lib\site-packages\datasets"); import load_dataset

from datasets import load_dataset
dataset = load_dataset("csebuetnlp/xnli_bn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BanglaBERT model
teacher_model = AutoModelForPreTraining.from_pretrained("csebuetnlp/banglabert").to(device)
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

train_dataloader = torch.utils.data.DataLoader(dataset = dataset['train'][2],
                                           batch_size = 32,
                                           shuffle = True)
# Define the student model
# class StudentModel(nn.Module):
#     def __init__(self, bert_model):
#         super(StudentModel, self).__init__()
#         self.bert = bert_model
#         self.classifier = nn.Linear(768, 2) # 2 classes in this example
        
#     def forward(self, input_ids):
#         _, pooled_output = self.bert(input_ids)
#         logits = self.classifier(pooled_output)
#         return logits

student_model = AutoModelForPreTraining.from_pretrained("csebuetnlp/banglabert_small")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert_small")
# Initialize the student model
# student_model = StudentModel(teacher_model).to(device)

# # Define the loss function
# def kd_loss(teacher_logits, student_logits, temperature=1.0):
#     # Cross-entropy loss between the teacher's soft labels and the student's predictions
#     kd_loss = nn.KLDivLoss(reduction='batchmean')(nn.Softmax(dim=-1)(student_logits/temperature), 
#                                                   nn.Softmax(dim=-1)(teacher_logits/temperature))
#     return kd_loss

# # Define the optimizer
# optimizer = optim.Adam(student_model.parameters(), lr=2e-5)

# # Define the temperature parameter
# temperature = 2.0
# num_epochs = 5
# # Train the student model
# for epoch in range(num_epochs):
#     for input_ids, attention_mask, labels in train_dataloader:
#         # Generate soft labels from the teacher model
#         with torch.no_grad():
#             teacher_logits = teacher_model(input_ids, attention_mask)[0]
            
#         # Forward pass through the student model
#         student_logits = student_model(input_ids, attention_mask)
        
#         # Compute the knowledge distillation loss
#         kd_loss = kd_loss(teacher_logits, student_logits, temperature)
        
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         kd_loss.backward()
#         optimizer.step()
        
#     print("Epoch {}: KD Loss = {}".format(epoch+1, kd_loss.item()))

# # Save the student model
# torch.save(student_model.state_dict(), 'path/to/student_model.pth')

# Define the loss function for knowledge distillation
def kd_loss(student_outputs, teacher_outputs, targets, temperature):
    # Calculate the soft targets
    soft_targets = nn.functional.log_softmax(teacher_outputs/temperature, dim=1)

    # Calculate the cross entropy loss
    kd_loss = nn.functional.kl_div(nn.functional.log_softmax(student_outputs/temperature, dim=1), soft_targets, reduction='batchmean') * temperature * temperature
    return kd_loss

# Initialize the student and teacher networks
# student_net = StudentModel(in_features=100, out_features=10)
# teacher_net = teacher_model(in_features=100, out_features=10)

# Load the data
inputs = torch.randn(64, 100)
targets = torch.randint(0, 10, (64,))

# Transfer the student and teacher networks to the GPU (if available)

student_net = student_model.to(device)
teacher_net = teacher_model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)

# Define the optimizer
optimizer = torch.optim.SGD(student_net.parameters(), lr=0.01)

# Train the student network
for epoch in range(10):
    # Forward pass
    student_outputs = student_net(inputs)
    teacher_outputs = teacher_net(inputs)

    # Calculate the loss
    loss = kd_loss(student_outputs, teacher_outputs, targets, temperature=4.0)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



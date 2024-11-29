from transformers import Trainer
import torch
from torch import nn

class DistilTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        student_model,
        temperature=2,
        lambda_param=0.5,
        *args,
        **kwargs
    ):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean", log_target=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, _, inputs, return_outputs=False, *args, **kwargs):
        student_output = self.student(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        soft_teacher = torch.nn.functional.log_softmax(
            teacher_output.logits / self.temperature, dim=-1
        )
        soft_student = torch.nn.functional.log_softmax(
            student_output.logits / self.temperature, dim=-1
        )
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (
            self.temperature**2
        )

        student_target_loss = student_output.loss

        loss = (
            1.0 - self.lambda_param
        ) * student_target_loss + self.lambda_param * distillation_loss

        return (loss, student_output) if return_outputs else loss
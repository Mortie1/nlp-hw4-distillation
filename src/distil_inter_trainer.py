from transformers import Trainer
import torch
from torch import nn


class DistilInterTrainer(Trainer):
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher.to(device)
        self.teacher.eval()

        self.student = student_model

        self.kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)

        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, _, inputs, return_outputs=False, *args, **kwargs):
        with torch.no_grad():
            teacher_output = self.teacher(
                **inputs, output_hidden_states=True, output_attentions=True
            )
        student_output = self.student(
            **inputs, output_hidden_states=True, output_attentions=True
        )

        soft_teacher = torch.nn.functional.log_softmax(
            teacher_output.logits / self.temperature, dim=-1
        )
        soft_student = torch.nn.functional.log_softmax(
            student_output.logits / self.temperature, dim=-1
        )
        distillation_loss = self.kldiv(soft_student, soft_teacher) * (
            self.temperature**2
        )

        student_target_loss = student_output.loss

        attn_loss = 0
        hidn_loss = 0
        for i in range(self.student.config.num_hidden_layers):
            attn_loss += torch.nn.functional.mse_loss(
                student_output.attentions[i], teacher_output.attentions[i * 2]
            )
            hidn_loss += torch.nn.functional.mse_loss(
                self.student.hidn_upscale[i](student_output.hidden_states[i]),
                teacher_output.hidden_states[i * 2],
            )

        loss = (
            1.0 - self.lambda_param
        ) * student_target_loss + self.lambda_param * distillation_loss + (1.0 - self.lambda_param) * (attn_loss + hidn_loss)

        student_output.hidden_states = None
        student_output.attentions = None
        return (loss, self.student(**inputs)) if return_outputs else loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments


class TrainingArgumentsDistill(TrainingArguments):
    """
    Contains extra training hyperparameters for the distillation loss function
    """
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class TrainerDistill(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        L = alpha*L_ce + (1-alpha)*L_kd
        L_kd = (T**2)*D_kl
        where:
            L_ce: cross entropy loss of student model and true label
            D_kl: KL divergence of student and teacher model logits
            L_kd: KL divergence normalized by squared temperature
        """
        # Get student cross entropy loss (L_ce) and logits
        labels = inputs.get("labels")
        outputs_student = model(**inputs)
        logits_student = outputs_student.logits

        # Cross entropy expects:
        # logits_student: (batch_size*seq_len, vocab_size)
        # labels: (batch_size*seq_len)
        loss_ce = F.cross_entropy(
            logits_student.view(-1, logits_student.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # Get teacher logits
        with torch.no_grad():
            output_teacher = self.teacher_model(**inputs)
            logits_teacher = output_teacher.logits

        loss_kd = self.forward_KL(logits_student, logits_teacher)
        loss = self.args.alpha * loss_ce + (1 - self.args.alpha) * loss_kd
        return (loss, outputs_student) if return_outputs else loss

    def forward_KL(self, student_logits, teacher_logits):
        """
        D_kl(P | Q) = sum(P(x)*log(P(x)/Q(x))
        P is the target distribution, Q is the approximation
        """
        # nn.KLDivLoss expects inputs as log probabilities and labels as normal
        # probabilities. Loss are averaged over batch dimension.
        kl_div = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * kl_div(
            F.log_softmax(student_logits / self.args.temperature, dim=-1),
            F.softmax(teacher_logits / self.args.temperature, dim=-1)
        )
        return loss_kd

    def reverse_KL(self, student_logits, teacher_logits):
        """
        D_kl(Q | P) = sum(Q(x)*log(Q(x)/P(x))
        P is the target distribution, Q is the approximation
        """
        # nn.KLDivLoss expects inputs as log probabilities and labels as normal
        # probabilities. Loss are averaged over batch dimension.
        kl_div = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * kl_div(
            F.log_softmax(teacher_logits / self.args.temperature, dim=-1),
            F.softmax(student_logits / self.args.temperature, dim=-1)
        )
        return loss_kd

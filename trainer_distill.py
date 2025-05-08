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
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Cross entropy expects:
        # logits_student: (batch_size*seq_len, vocab_size)
        # labels: (batch_size*seq_len)
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        loss_kd = self.forward_KL(student_logits, teacher_logits)
        loss = self.args.alpha * loss_ce + (1 - self.args.alpha) * loss_kd
        return (loss, student_outputs) if return_outputs else loss

    def adaptive_kl_loss(self, student_logits, teacher_logits, temperature=1.0, mu=0.5):
        prob_teacher = F.softmax(teacher_logits / temperature, dim=-1)

        # Max value over vocab for each sequence
        max_probs_teacher, _ = prob_teacher.max(dim=-1)     # [batch_size, seq_len]
        flat_max_probs = max_probs_teacher.view(dim=-1)     # [batch_size * seq_len]
        # Define head tokens (top mu% by teacher confidence)
        threshold = torch.quantile(flat_max_probs, 1 - mu)
        head_mask = (max_probs_teacher >= threshold).float()  # 1 for head, 0 for tail

        fwd_kl = self.forward_KL(student_logits, teacher_logits)
        rev_kl = self.reverse_KL(student_logits, teacher_logits)

        # Compute gaps
        g_head = (head_mask * fwd_kl).sum()
        g_tail = ((1 - head_mask) * rev_kl).sum()
        g_total = g_head + g_tail + 1e-8  # avoid division by zero

        # Weighted AKL loss
        akl_loss = (g_head / g_total) * fwd_kl.mean() + (g_tail / g_total) * rev_kl.mean()

        return akl_loss

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

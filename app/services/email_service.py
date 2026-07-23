import resend
from app.config import settings
import logging

logger = logging.getLogger(__name__)

resend.api_key = settings.RESEND_API_KEY

def send_welcome_email(to_email: str, username: str):
    if not settings.RESEND_API_KEY:
        logger.warning("RESEND_API_KEY is not set. Skipping welcome email.")
        return

    try:
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                <h2 style="color: #4F46E5;">Selamat datang di Synapsa, {username}! 🧠</h2>
                <p>Kami sangat senang Anda bergabung. Platform pembelajaran adaptif ini siap membantu Anda mencapai potensi belajar maksimal.</p>
                <p>Mulai perjalanan belajar Anda sekarang!</p>
                <br/>
                <p>Salam hangat,<br/>Tim Synapsa</p>
            </body>
        </html>
        """
        params: resend.Emails.SendParams = {
            "from": settings.MAIL_FROM,
            "to": [to_email],
            "subject": "Selamat datang di Synapsa!",
            "html": html_content,
        }
        email = resend.Emails.send(params)
        logger.info(f"Welcome email sent to {to_email}. ID: {email.get('id')}")
    except Exception as e:
        logger.error(f"Failed to send welcome email to {to_email}: {str(e)}")

def send_verification_email(to_email: str, username: str, code: str):
    if not settings.RESEND_API_KEY:
        logger.warning("RESEND_API_KEY is not set. Skipping verification email.")
        return

    try:
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                <h2 style="color: #4F46E5;">Verifikasi Email Anda</h2>
                <p>Halo {username},</p>
                <p>Gunakan kode berikut untuk memverifikasi akun Synapsa Anda:</p>
                <div style="background: #f3f4f6; border-radius: 12px; padding: 20px; text-align: center; margin: 20px 0;">
                    <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #4F46E5;">{code}</span>
                </div>
                <p style="color: #666; font-size: 14px;">Kode ini berlaku selama 15 menit.</p>
                <p>Jika Anda tidak mendaftar di Synapsa, abaikan email ini.</p>
                <br/>
                <p>Salam hangat,<br/>Tim Synapsa</p>
            </body>
        </html>
        """
        params: resend.Emails.SendParams = {
            "from": settings.MAIL_FROM,
            "to": [to_email],
            "subject": "Kode Verifikasi Synapsa",
            "html": html_content,
        }
        email = resend.Emails.send(params)
        logger.info(f"Verification email sent to {to_email}. ID: {email.get('id')}")
    except Exception as e:
        logger.error(f"Failed to send verification email to {to_email}: {str(e)}")

def send_password_reset_email(to_email: str, username: str, reset_url: str):
    if not settings.RESEND_API_KEY:
        logger.warning("RESEND_API_KEY is not set. Skipping password reset email.")
        return

    try:
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                <h2 style="color: #4F46E5;">Reset Password</h2>
                <p>Halo {username},</p>
                <p>Kami menerima permintaan untuk mereset password akun Synapsa Anda.</p>
                <p>Klik tombol di bawah untuk membuat password baru:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_url}" style="background: #4F46E5; color: white; padding: 14px 28px; border-radius: 10px; text-decoration: none; font-weight: bold; display: inline-block;">Reset Password</a>
                </div>
                <p style="color: #666; font-size: 14px;">Link ini berlaku selama 1 jam.</p>
                <p>Jika Anda tidak meminta reset password, abaikan email ini.</p>
                <br/>
                <p>Salam hangat,<br/>Tim Synapsa</p>
            </body>
        </html>
        """
        params: resend.Emails.SendParams = {
            "from": settings.MAIL_FROM,
            "to": [to_email],
            "subject": "Reset Password Synapsa",
            "html": html_content,
        }
        email = resend.Emails.send(params)
        logger.info(f"Password reset email sent to {to_email}. ID: {email.get('id')}")
    except Exception as e:
        logger.error(f"Failed to send password reset email to {to_email}: {str(e)}")

def send_daily_reminder_email(to_email: str, username: str):
    if not settings.RESEND_API_KEY:
        logger.warning("RESEND_API_KEY is not set. Skipping reminder email.")
        return

    try:
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                <h2 style="color: #F59E0B;">Halo {username}, jangan putus streak belajar Anda! 🔥</h2>
                <p>Kami melihat Anda belum login hari ini. Luangkan sedikit waktu untuk melanjutkan progress belajar Anda.</p>
                <p>Setiap menit sangat berarti untuk perkembangan Anda!</p>
                <br/>
                <p>Salam hangat,<br/>Tim Synapsa</p>
            </body>
        </html>
        """
        params: resend.Emails.SendParams = {
            "from": settings.MAIL_FROM,
            "to": [to_email],
            "subject": "Waktunya belajar di Synapsa!",
            "html": html_content,
        }
        email = resend.Emails.send(params)
        logger.info(f"Reminder email sent to {to_email}. ID: {email.get('id')}")
    except Exception as e:
        logger.error(f"Failed to send reminder email to {to_email}: {str(e)}")

def send_progress_email(to_email: str, username: str, topic: str):
    if not settings.RESEND_API_KEY:
        logger.warning("RESEND_API_KEY is not set. Skipping progress email.")
        return

    try:
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                <h2 style="color: #10B981;">Kerja bagus, {username}! 🚀</h2>
                <p>Anda baru saja menyelesaikan bagian pembelajaran: <strong>{topic}</strong>.</p>
                <p>Terus pertahankan semangat belajar Anda, kami bangga dengan progress Anda!</p>
                <br/>
                <p>Salam hangat,<br/>Tim Synapsa</p>
            </body>
        </html>
        """
        params: resend.Emails.SendParams = {
            "from": settings.MAIL_FROM,
            "to": [to_email],
            "subject": f"Progress Belajar: {topic} Selesai!",
            "html": html_content,
        }
        email = resend.Emails.send(params)
        logger.info(f"Progress email sent to {to_email}. ID: {email.get('id')}")
    except Exception as e:
        logger.error(f"Failed to send progress email to {to_email}: {str(e)}")

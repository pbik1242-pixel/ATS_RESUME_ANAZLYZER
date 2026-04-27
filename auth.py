from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
import database as db_mod
import requests
import secrets
from urllib.parse import urlencode
from werkzeug.security import check_password_hash

from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    GOOGLE_REQUIRE_VERIFIED_EMAIL,
    RECAPTCHA_MODE,
    RECAPTCHA_SECRET_KEY,
    RECAPTCHA_SITE_KEY,
)

router = APIRouter()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"


def _login_context(request: Request, hr_mode: bool, error: str | None = None, message: str | None = None) -> dict:
    from app import base_context

    recaptcha_enabled = RECAPTCHA_MODE == "google" and bool(RECAPTCHA_SITE_KEY and RECAPTCHA_SECRET_KEY)
    google_enabled = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
    google_start_url = f"/auth/google/start?hr=1" if hr_mode else "/auth/google/start"
    return base_context(
        request,
        hr_mode=hr_mode,
        error=error,
        message=message,
        recaptcha_site_key=RECAPTCHA_SITE_KEY if recaptcha_enabled else "",
        recaptcha_enabled=recaptcha_enabled,
        google_login_enabled=google_enabled,
        google_login_url=google_start_url if google_enabled else None,
    )


def _verify_recaptcha(token: str | None, remote_ip: str | None = None) -> tuple[bool, str | None]:
    if not token:
        return False, "Captcha required"

    if not (RECAPTCHA_SITE_KEY and RECAPTCHA_SECRET_KEY):
        return True, None

    payload: dict[str, str] = {
        "secret": RECAPTCHA_SECRET_KEY,
        "response": token,
    }
    if remote_ip:
        payload["remoteip"] = remote_ip

    try:
        response = requests.post(RECAPTCHA_VERIFY_URL, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return False, "Captcha verification failed"

    if data.get("success"):
        return True, None
    return False, "Captcha verification failed"


def _google_redirect_uri(request: Request) -> str:
    host = (request.url.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return str(request.url_for("google_oauth_callback"))
    if GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    return str(request.url_for("google_oauth_callback"))


@router.get("/login")
async def login_page(request: Request):
    from app import templates

    return templates.TemplateResponse(request, "login.html", _login_context(request, hr_mode=False))


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    captcha: str | None = Form(None),
    g_recaptcha_response: str | None = Form(None, alias="g-recaptcha-response"),
):
    from app import set_session_user, templates

    token = g_recaptcha_response or captcha
    remote_ip = request.client.host if request.client else None
    captcha_ok, captcha_error = _verify_recaptcha(token, remote_ip)
    if not captcha_ok:
        return templates.TemplateResponse(request, "login.html", _login_context(request, hr_mode=False, error=captcha_error), status_code=400)

    user = db_mod.get_user_by_email(email)
    if user and check_password_hash(user["password"], password):
        set_session_user(request, user, remember_me=False)
        client_host = request.client.host if request.client else ""
        db_mod.record_login_success(user["id"], user["email"], client_host, request.headers.get("user-agent", ""))
        if not request.session.get("resume_consent"):
            return RedirectResponse("/consent", status_code=303)
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(request, "login.html", _login_context(request, hr_mode=False, error="Invalid credentials"), status_code=400)


@router.get("/hr/login")
async def hr_login_page(request: Request):
    from app import current_user, is_hr_user, templates

    user = current_user(request)
    if user and is_hr_user(user):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", _login_context(request, hr_mode=True))


@router.post("/hr/login")
async def hr_login_submit(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    captcha: str | None = Form(None),
    g_recaptcha_response: str | None = Form(None, alias="g-recaptcha-response"),
):
    from app import is_admin_user, is_hr_user, set_session_user, templates

    token = g_recaptcha_response or captcha
    remote_ip = request.client.host if request.client else None
    captcha_ok, captcha_error = _verify_recaptcha(token, remote_ip)
    if not captcha_ok:
        return templates.TemplateResponse(request, "login.html", _login_context(request, hr_mode=True, error=captcha_error), status_code=400)

    user = db_mod.get_user_by_email(email)
    if not user or not check_password_hash(user["password"], password):
        return templates.TemplateResponse(request, "login.html", _login_context(request, hr_mode=True, error="Invalid credentials"), status_code=400)

    if not (is_hr_user(user) or is_admin_user(user)):
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=True, error="This account does not have HR access."),
            status_code=403,
        )

    set_session_user(request, user, remember_me=False)
    client_host = request.client.host if request.client else ""
    db_mod.record_login_success(user["id"], user["email"], client_host, request.headers.get("user-agent", ""))
    if not request.session.get("resume_consent"):
        return RedirectResponse("/consent", status_code=303)
    return RedirectResponse("/", status_code=303)


@router.get("/auth/google/start")
async def google_oauth_start(request: Request, hr: int = 0):
    from app import templates

    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=bool(hr), error="Google sign-in is not configured yet."),
            status_code=400,
        )

    state = secrets.token_urlsafe(24)
    request.session["google_oauth_state"] = state
    request.session["google_login_mode"] = "hr" if hr else "candidate"

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": _google_redirect_uri(request),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=303)


@router.get("/auth/google/callback", name="google_oauth_callback")
async def google_oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    from app import is_admin_user, is_hr_user, set_session_user, templates

    login_mode = request.session.get("google_login_mode", "candidate")
    hr_mode = login_mode == "hr"

    if error:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=hr_mode, error=f"Google sign-in failed: {error}"),
            status_code=400,
        )

    expected_state = request.session.get("google_oauth_state")
    if not code or not state or state != expected_state:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=hr_mode, error="Google sign-in could not be verified."),
            status_code=400,
        )

    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=hr_mode, error="Google sign-in is not configured yet."),
            status_code=400,
        )

    try:
        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": _google_redirect_uri(request),
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        token_response.raise_for_status()
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("No access token returned by Google")

        profile_response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
    except Exception:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=hr_mode, error="Google sign-in failed while contacting Google."),
            status_code=400,
        )

    email = str(profile.get("email") or "").strip().lower()
    if not email:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=hr_mode, error="Google did not share an email address."),
            status_code=400,
        )

    email_verified = profile.get("email_verified")
    if isinstance(email_verified, str):
        email_verified = email_verified.strip().lower() == "true"
    if GOOGLE_REQUIRE_VERIFIED_EMAIL and not email_verified:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=hr_mode, error="Your Google email is not verified."),
            status_code=400,
        )

    name = str(profile.get("name") or profile.get("given_name") or email.split("@", 1)[0]).strip()
    user = db_mod.get_user_by_email(email)

    if not user:
        if hr_mode:
            request.session.pop("google_oauth_state", None)
            request.session.pop("google_login_mode", None)
            return templates.TemplateResponse(
                request,
                "login.html",
                _login_context(request, hr_mode=True, error="No HR account is linked to that Google email."),
                status_code=403,
            )
        user = db_mod.ensure_google_user(email, name, role="candidate")
    elif hr_mode and not (is_hr_user(user) or is_admin_user(user)):
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=True, error="This Google account does not have HR access."),
            status_code=403,
        )
    elif name and not str(user["name"] or "").strip():
        user = db_mod.ensure_google_user(email, name, role=str(user["role"] or "candidate"))

    if not user:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_login_mode", None)
        return templates.TemplateResponse(
            request,
            "login.html",
            _login_context(request, hr_mode=hr_mode, error="Google sign-in could not create your account."),
            status_code=400,
        )

    request.session.pop("google_oauth_state", None)
    request.session.pop("google_login_mode", None)

    set_session_user(request, user, remember_me=False)
    client_host = request.client.host if request.client else ""
    db_mod.record_login_success(user["id"], user["email"], client_host, request.headers.get("user-agent", ""))
    if not request.session.get("resume_consent"):
        return RedirectResponse("/consent", status_code=303)
    return RedirectResponse("/", status_code=303)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@router.get("/register")
async def register_page(request: Request):
    from app import templates

    return templates.TemplateResponse(request, "register.html", _login_context(request, hr_mode=False))


@router.post("/register")
async def register_submit(
    request: Request,
    name: str = Form(""),
    email: str = Form(""),
    password: str = Form(""),
    confirm_password: str = Form(""),
    role: str = Form("candidate"),
):
    from app import templates

    ok, message = db_mod.register_user(name, email, password, confirm_password, role)
    if not ok:
        return templates.TemplateResponse(request, "register.html", _login_context(request, hr_mode=False, error=message), status_code=400)
    return RedirectResponse("/login", status_code=303)


@router.get("/hr/register")
async def hr_register_page(request: Request):
    from app import templates

    return templates.TemplateResponse(request, "register.html", _login_context(request, hr_mode=True))


@router.post("/hr/register")
async def hr_register_submit(
    request: Request,
    name: str = Form(""),
    email: str = Form(""),
    password: str = Form(""),
    confirm_password: str = Form(""),
):
    from app import templates

    ok, message = db_mod.register_user(name, email, password, confirm_password, "hr")
    if not ok:
        return templates.TemplateResponse(request, "register.html", _login_context(request, hr_mode=True, error=message), status_code=400)
    return RedirectResponse("/hr/login", status_code=303)

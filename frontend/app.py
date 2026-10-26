"""Streamlit UI for the recipe agent."""

import os

import httpx
import streamlit as st

from utils.pdf import generate_plan_pdf


def _get_config(key: str, default: str) -> str:
    """Resolve configs"""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


API_BASE = _get_config("API_BASE_URL", "http://localhost:8000").rstrip("/")
PLAN_REQUEST_TIMEOUT = float(_get_config("PLAN_REQUEST_TIMEOUT", "120.0"))
REFINE_REQUEST_TIMEOUT = float(_get_config("REFINE_REQUEST_TIMEOUT", "60.0"))


def _parse_csv(raw: str) -> list[str]:
    """Split a comma-separated field into trimmed, non-empty items."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _render_sidebar() -> dict | None:
    """Render the profile form; return the request payload on submit, else None."""
    st.sidebar.title("Your Profile")

    api_key = st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        key="openai_api_key",
        help=(
            "Required. Used only for this session to generate your plan and "
            "never stored on the server."
        ),
    )

    with st.sidebar.form("profile_form"):
        age = st.number_input("Age", min_value=18, max_value=120, value=30, step=1)
        height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=175, step=1)
        weight_kg = st.number_input("Weight (kg)", min_value=35, max_value=300, value=75, step=1)
        sex = st.selectbox("Sex", ["male", "female"])
        goal = st.selectbox("Goal", ["lose", "maintain", "gain"], index=1)
        meals_per_day = st.number_input("Meals per day", min_value=1, max_value=4, value=3, step=1)
        days = st.number_input("Days", min_value=1, max_value=14, value=7, step=1)
        allergies = st.text_input("Allergies (comma-separated)")
        dislikes = st.text_input("Dislikes (comma-separated)")
        submitted = st.form_submit_button("Build my plan")

    if not submitted:
        return None

    if not api_key.strip():
        st.sidebar.error("Please enter your OpenAI API key to build a plan.")
        return None

    return {
        "age": int(age),
        "height_cm": int(height_cm),
        "weight_kg": int(weight_kg),
        "sex": sex,
        "goal": goal,
        "meals_per_day": int(meals_per_day),
        "days": int(days),
        "allergies": _parse_csv(allergies),
        "dislikes": _parse_csv(dislikes),
        "openai_api_key": api_key.strip(),
    }


def _request_plan(payload: dict) -> None:
    """POST the profile to the backend and cache the plan in session state."""
    with st.spinner("Finding recipes..."):
        try:
            response = httpx.post(
                f"{API_BASE}/plan", json=payload, timeout=PLAN_REQUEST_TIMEOUT
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except ValueError:
                detail = exc.response.text
            st.error(f"The planner returned an error ({exc.response.status_code}): {detail}")
            return
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the planner at {API_BASE}: {exc}")
            return

    data = response.json()
    st.session_state["plan"] = data
    st.session_state["messages"] = data.get("messages", [])
    st.session_state["profile_id"] = data.get("profile_id")


def _render_plan() -> None:
    """Render the cached plan, or a placeholder when there is none."""
    plan = st.session_state.get("plan")
    if plan is None:
        st.info("Fill in your profile and click the \"Build my plan\" button.")
        return

    targets = plan.get("targets") or {}
    if targets:
        st.caption(
            f"Daily targets: {targets.get('calories', '?')} kcal · "
            f"{targets.get('protein_g', '?')}g protein · "
            f"{targets.get('carbs_g', '?')}g carbs · "
            f"{targets.get('fat_g', '?')}g fat"
        )

    for day_name, day in plan["days"].items():
        with st.expander(day_name):
            for meal, recipe in day["meals"].items():
                st.markdown(
                    f"**{meal}**: [{recipe['title']}]({recipe['url']}) "
                    f"— {recipe['calories']} kcal"
                )


def _render_followup() -> None:
    """Render the follow-up chat box (only once a plan exists)."""
    plan = st.session_state.get("plan")
    if plan is None:
        return

    st.subheader("Follow-up")

    if st.session_state.get("plan_done"):
        st.success("Plan saved!")
        return

    profile_id = st.session_state.get("profile_id")
    if profile_id is None:
        st.info("This plan has no saved profile yet, so follow-ups are unavailable.")
        return

    api_key = st.session_state.get("openai_api_key", "").strip()
    user_input = st.text_input("Ask a follow-up...")
    if st.button("Send") and user_input:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar to continue.")
            return
        try:
            response = httpx.post(
                f"{API_BASE}/refine",
                json={
                    "message": user_input,
                    "profile_id": profile_id,
                    "messages": st.session_state.get("messages", []),
                    "openai_api_key": api_key,
                },
                timeout=REFINE_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            st.error(f"Refine request failed: {exc}")
            return

        data = response.json()
        st.session_state["messages"] = data["messages"]
        st.session_state["plan"] = data["plan"]
        st.markdown(data.get("message", ""))
        if data.get("done"):
            st.session_state["plan_done"] = True
            st.rerun()

def _render_action_buttons() -> None:
    """Display a PDF download button when a meal plan exists."""
    plan = st.session_state.get("plan")

    if plan is None:
        return

    try:
        pdf_bytes = generate_plan_pdf(plan)
    except (ValueError, TypeError) as exc:
        st.error(f"Could not generate PDF: {exc}")
        return

    with st.container(horizontal=True, horizontal_alignment="right"):
        st.download_button(
            label="Download as PDF",
            type="primary",
            data=pdf_bytes,
            file_name="meal_plan.pdf",
            mime="application/pdf",
            icon=":material/download:",
            on_click="ignore",
            width="content",
        )

def _render_footer() -> None:
    """Render a trust-building footer with author and source links."""
    st.divider()
    st.markdown(
        '<div style="text-align: center; color: gray;">'
        'Created with ❤️ by '
        '<a href="https://github.com/haingo-raz" target="_blank" '
        'rel="noopener noreferrer">haingo-raz</a>'
        ' · Codebase available on '
        '<a href="https://github.com/haingo-raz/ai_planner/" target="_blank" '
        'rel="noopener noreferrer">GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )

def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Recipe Discovery App", page_icon="🥗")
    st.title("Recipe Discovery App")

    payload = _render_sidebar()
    if payload is not None:
        _request_plan(payload)

    _render_plan()
    # _render_followup() temporarily disabled
    _render_action_buttons()
    _render_footer()


if __name__ == "__main__":
    main()

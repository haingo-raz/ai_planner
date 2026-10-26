"""Create downloadable meal-plan PDFs from the Streamlit API response."""

from html import escape
from io import BytesIO
from typing import Any, Mapping
from urllib.parse import urlsplit

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


def _safe_recipe_url(value: object) -> str | None:
    """Only expose HTTP(S) links with a hostname in generated PDFs."""
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
    except ValueError:
        return None
    return value.strip()


def generate_plan_pdf(plan: Mapping[str, Any]) -> bytes:
    """Return a PDF for a serialized /plan or /refine response."""
    days = plan.get("days")
    if not isinstance(days, Mapping) or not days:
        raise ValueError("The meal plan must contain at least one day.")

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Meal Plan",
        author="Recipe Discovery Agent",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="MealPlanTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=20, leading=25, spaceAfter=18,
        textColor=colors.HexColor("#256D45"),
    ))
    styles.add(ParagraphStyle(
        name="MealPlanDay", parent=styles["Heading2"],
        fontSize=14, leading=18, spaceBefore=14, spaceAfter=8,
        textColor=colors.HexColor("#256D45"),
    ))
    styles.add(ParagraphStyle(
        name="MealPlanBody", parent=styles["BodyText"],
        fontSize=10, leading=15, spaceAfter=5,
        wordWrap="CJK",
    ))

    story = [Paragraph("Your Meal Plan", styles["MealPlanTitle"])]
    targets = plan.get("targets")
    if isinstance(targets, Mapping) and targets:
        def target(key: str) -> str:
            return escape(str(targets.get(key, "?")))

        story.append(Paragraph("Daily Nutritional Targets", styles["Heading3"]))
        story.append(Paragraph(
            f"Calories: {target('calories')} kcal | "
            f"Protein: {target('protein_g')} g | "
            f"Carbs: {target('carbs_g')} g | "
            f"Fat: {target('fat_g')} g",
            styles["MealPlanBody"],
        ))
        story.append(Spacer(1, 12))

    for day_name, day in days.items():
        if not isinstance(day, Mapping):
            raise ValueError(f"Invalid day: {day_name}")
        meals = day.get("meals")
        if not isinstance(meals, Mapping):
            raise ValueError(f"Missing meals for day: {day_name}")

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Paragraph(escape(str(day_name)), styles["MealPlanDay"]))
        for meal_name, recipe in meals.items():
            if not isinstance(recipe, Mapping):
                raise ValueError(f"Invalid recipe in {day_name}: {meal_name}")
            title = escape(str(recipe.get("title", "Untitled recipe")))
            calories = escape(str(recipe.get("calories", "?")))
            meal_label = escape(str(meal_name).replace("_", " ").title())
            block = [Paragraph(
                f"<b>{meal_label}</b>: {title} ({calories} kcal)",
                styles["MealPlanBody"],
            )]
            url = _safe_recipe_url(recipe.get("url"))
            if url:
                block.append(Paragraph(
                    f'<link href="{escape(url, quote=True)}" color="#1769AA">'
                    "View recipe</link>",
                    styles["MealPlanBody"],
                ))
            block.append(Spacer(1, 7))
            story.append(KeepTogether(block))

    document.build(story)
    return buffer.getvalue()

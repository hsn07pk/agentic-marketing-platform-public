import sys
import os
from pathlib import Path
sys.path.insert(0, '/app' if os.path.exists('/app/src') else str(Path(__file__).parent.parent))

from src.data_layer.database.connection import sync_session_maker
from src.data_layer.database.models import AgentAction
from sqlalchemy import select

with sync_session_maker() as session:
    actions = session.execute(select(AgentAction).limit(20)).scalars().all()
    print(f"Found {len(actions)} actions.")
    for a in actions:
        start = a.started_at.isoformat() if a.started_at else "None"
        end = a.completed_at.isoformat() if hasattr(a, 'completed_at') and a.completed_at else "None"
        dur_ms = a.duration_ms if hasattr(a, 'duration_ms') else "N/A"
        output_dur = a.output_data.get('duration') if a.output_data else "N/A"
        print(f"ID: {a.id} | Start: {start} | End: {end} | DurMS: {dur_ms} | OutDur: {output_dur}")

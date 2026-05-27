"""Fix the corrupted _load_prerace_from_clickhouse function."""
import re

filepath = r"c:\telemetry-producer\f1_predictions\inference_engine.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Normalize to LF for matching
content = content.replace("\r\n", "\n")

# The corrupted block: "except Exception:" followed by orphaned SQL
old_block = """    except Exception:
                WHERE event   = '{event}'
                  AND year    = {year}
                  AND session = '{session}'
                \"\"\"
            )
            if not total_df.empty and pd.notna(total_df.iloc[0]["max_lap"]):
                total_laps = int(total_df.iloc[0]["max_lap"])
        except Exception:
            pass

    return pre_race_ctx, num_to_code, total_laps"""

new_block = """    except Exception:
        pass

    # Re-resolve any pre_race_ctx entries that are still keyed by driver number
    for raw_num, code in list(num_to_code.items()):
        if raw_num in pre_race_ctx and code != raw_num:
            pre_race_ctx[code] = pre_race_ctx.pop(raw_num)
            pre_race_ctx[code]["driver_code"] = code

    # -- Total laps -------------------------------------------------------
    total_laps = RACE_LAPS.get(event, 0)
    if total_laps == 0:
        try:
            total_df = ch.query_df(
                f\"\"\"
                SELECT max(LapNumber) AS max_lap
                FROM raw_telemetry
                WHERE event   = '{event}'
                  AND year    = {year}
                  AND session = '{session}'
                \"\"\"
            )
            if not total_df.empty and pd.notna(total_df.iloc[0]["max_lap"]):
                total_laps = int(total_df.iloc[0]["max_lap"])
        except Exception:
            pass

    return pre_race_ctx, num_to_code, total_laps"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS - corrupted block replaced")
else:
    # Debug: find the marker
    idx = content.find('except Exception:\n                WHERE event')
    if idx >= 0:
        print(f"Found marker at index {idx}")
        print("Context:")
        print(repr(content[idx:idx+300]))
    else:
        print("Marker not found - checking for already fixed...")
        if "Re-resolve any pre_race_ctx" in content:
            print("Already fixed!")
        else:
            print("FAILED - pattern not found anywhere")

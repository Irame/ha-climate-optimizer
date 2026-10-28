# Climate optimizer

Decides when the AC units and heaters of an apartment run, so that heating (or cooling) costs as little as possible for the comfort you ask for. It learns how the apartment holds and loses heat from Home Assistant's history, plans the next 12 to 36 hours against the electricity prices and the weather forecast, and tells each [proxy climate](https://github.com/Irame/ha-proxy-climate) entity in `auto` mode what to do, once a minute.

It runs as a container next to Home Assistant, with its own web page for the plan, the history, the comfort settings and the model.

## How it decides

**The model.** Every room is three heat stores: its air (what the thermometer reads), its surfaces and furniture (charged within hours), and the structure behind them (walls and floors: days). Rooms exchange heat through their air. Losses go to the outside and to the neighbours; sun comes in through the facades; open windows and doors add losses. Each AC unit delivers a learned amount of heat per fan level, a little less the colder it is outside, and all units share one outdoor unit with a learned efficiency. The numbers are fitted every night to the recorded temperatures, unit states, plug and phase power, and checked on days held back from the fit. Two weeks of history cannot tell every number apart (a bigger heat store or a smaller loss look much alike for a while), so the fit leans on what a typical apartment is like and lets the data move away from it where it clearly says so; the units' heat per fan level comes from the phase measurement. A thermometer that stands still for hours counts as having stopped reporting. Sunny days teach the model about sun; until it has seen a few, it predicts them too cool.

**The plan.** Every five minutes the optimizer minimises, over the horizon of known prices:

> electricity × price + comfort cost of every room + compressor starts + the value of heat missing at the end

subject to the model, each unit's lowest and highest fan level, the outdoor unit's shared capacity, minimum run and pause times, and the times a unit must not run. It is a mixed-integer linear programme, solved with HiGHS in about a second. The plan heats ahead of expensive hours, lets rooms drift where a deviation costs less than the electricity saved, and keeps compressor starts few.

**Comfort costs.** Being off the comfort temperature costs money, per kelvin and hour, separately for too cold and too warm: a deadband that costs nothing, a linear and a quadratic part, and a limit that is not crossed unless it can't be helped. Four functions come predefined (standard, relaxed, strict, frost protection); any number can be added. The comfort temperature itself is the proxy's target temperature.

**Schedules.** Rules switch a room to another function, move its comfort temperature, or keep units off, at times that recur by an iCalendar rule. "Every other week, Monday to Thursday, 18:00 to 23:00" is weekly, every 2 weeks, those days, from 18:00 to 23:00. Rules have priorities. Events in a Home Assistant calendar override them for single occasions, by their title:

| Title | Effect |
|---|---|
| `Wohnzimmer: Relaxed` | the living room uses the relaxed function |
| `Schlafzimmer: aus` | the bedroom's units stay off |
| `Alle: +1` | every comfort temperature 1 K higher |

**Fan levels.** The plan gives each unit a heat output for the current five minutes. A fan level delivers fixed heat, so a target between two levels is met by alternating between them minute by minute.

**Heating or cooling.** All AC units share one outdoor unit, so they run in the same mode. When rooms would drift too warm and it gets warm enough outside, cooling is planned too and the cheaper plan wins.

## Running it

```sh
cp docker-compose.example.yml docker-compose.yml   # set HA_URL, and HA_TOKEN in .env
docker compose up -d --build
```

Then open `http://<host>:8099`:

1. **Settings:** check the Home Assistant connection. For every room, choose the thermometer and the window or door sensors, and tick where its windows face: the model then looks for sun only from there, which it learns much sooner. For every unit, choose its proxy entity and the device under it, plus the fan modes quietest first. Then set the outdoor thermometer, the weather entity, the phase the outdoor unit hangs on, and the price source. Save.
2. The optimizer backfills what Home Assistant's recorder has (10 days by default) and trains on it. This takes a minute or two. The **Model** page shows how well it predicts.
3. Point every proxy at the optimizer and set it to `auto`:

   ```yaml
   optimizer_url: http://<host>:8099
   auto_fallback: heat
   ```

4. The optimizer starts in **shadow mode**: it plans and shows what it would do, and the proxies keep their own thermostats. When the plans look right, switch shadow mode off in Settings.

Without Home Assistant, `DEMO=1` runs the optimizer against a simulated apartment with a week of history:

```sh
docker run --rm -p 8099:8099 -e DEMO=1 $(docker build -q .)
```

The container starts as root, hands its data directory to `PUID:PGID` (default `1000:1000`) and then runs as that user, so a `./data` directory that Docker created as root works as it is. With `user:` set in the compose file the container cannot change ownership; the directory must then already be writable for that user.

### Diagnostics

**Download diagnostics** on the Model page writes one file with what the training reads: the recorded temperatures, unit states and power (numeric sensors as 5-minute means), the settings without the Home Assistant URL and tokens, the trained model and its report, and the last three days of plans. Without the page:

```sh
docker exec climate-optimizer python -m climate_optimizer.diagnostics --out /data/diagnostics.json.gz
```

The file is an export: Settings, "Load an export" reads it back in.

### Prices

- `tibber_prices`: the `tibber_prices.get_price` action of the Tibber Prices integration.
- `a sensor's attributes`: any sensor that lists coming prices in an attribute, such as Nord Pool, EPEX Spot or Tibber. Any list of entries with a start time and a price is understood. Cents are recognised.

Beyond the known prices, the plan assumes the same price as 24 hours earlier.

### Security

The API and the page are open on the network unless an API token is set in Settings. With one set, proxies send it as `optimizer_token`, and the page asks for it once.

## For the proxies: `POST /api/v1/command`

The proxy sends its state (see the proxy's README). The answer:

```json
{"hvac_mode": "heat", "fan_level": 2, "valid_until": "2026-10-27T10:05:00+00:00", "reason": "heat 2.1 kW · 24.3 ct/kWh"}
```

`hvac_mode` is `heat`, `cool` or `off`, and `fan_level` indexes the proxy's `source_fan_modes`. The answer is valid for three minutes; after that the proxy falls back to its own thermostat. In shadow mode, and for proxies the optimizer doesn't know, the answer is `{"follow": false}`.

The proxy's fan mode offset limits the levels: an offset of −1 caps the top level one lower, and +1 raises the lowest level by one.

## Development

```sh
pip install -e ".[test]"
pytest
DEMO=1 DATA_DIR=. python -m climate_optimizer   # data in ./demo-data
```

| Module | Contents |
|---|---|
| `climate_optimizer/model/` | the thermal model, the training grid, the fit, the synthetic apartment |
| `mpc.py` | the planning programme |
| `comfort.py` | comfort functions and rules |
| `dispatch.py` | fan levels minute by minute |
| `planner.py` | gathering inputs, planning, answering the proxies |
| `collector.py`, `ha.py`, `store.py` | the Home Assistant connection and the recorded history |
| `app.py`, `web/` | the API and the page |

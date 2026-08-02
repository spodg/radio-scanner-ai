# BCD436HP Serial Commands Reference

The BCD436HP communicates via USB serial at 115200 baud. Commands are ASCII
strings terminated with `\r` (carriage return).

## Verified Working Commands

| Command | Response | Purpose |
|---------|----------|---------|
| `MDL` | `MDL,BCD436HP` | Get scanner model |
| `VER` | `VER,Version X.XX.XX` | Get firmware version |
| `SQL` | `SQL,2` | Get squelch level (0-15) |
| `SQL,5` | `SQL,OK` | Set squelch level |
| `VOL` | `VOL,18` | Get volume (0-29) |
| `VOL,10` | `VOL,OK` | Set volume |
| `GLG` | See below | Get current reception status |
| `STS` | See below | Get full LCD screen content |
| `PWR` | `PWR,signal,freq` | Get signal strength |
| `KEY,x,P` | `KEY,OK` | Simulate key press |
| `JPM,SCN_MODE` | `JPM,OK` | Jump to scan mode |
| `JPM,CC_MODE` | `JPM,OK` | Jump to close call mode |

## GLG (Get Current Reception)

Returns current frequency, system, group, channel, and squelch state.

```
GLG
GLG,0453.3500,NFM,0,212,Suffolk,City of Boston,Housing Police,1,0,,,NONE
     │         │   │ │   │       │              │             │ │
     freq      mod att ?  system  group          channel      sql mut
```

Fields:
- `frequency` — current frequency or TGID
- `modulation` — NFM, FM, AM, P25, DMR, etc.
- `attenuation` — 0=off, 1=on
- `system` — system name
- `group` — group/department name
- `channel` — channel name
- `squelch` — 1=open (receiving), 0=closed (scanning)
- `mute` — 1=muted, 0=unmuted

## STS (Screen Status)

Returns the full LCD display content with section separators.

```
STS
STS,flags, F0:header1 ,, S0:header2 ,________________________,
section1_line1,,section1_line2,,section1_line3,________________________,
section2_line1,,section2_line2,,section2_line3,________________________,
section3_line1,,section3_line2,,section3_line3,________________________,
footer,,tag_info,,flags
```

The `________________________` (24 underscores) separates display sections.

## KEY (Simulate Key Press)

| Key Code | Physical Key | Use |
|----------|-------------|-----|
| `KEY,M,P` | MENU | Open menu |
| `KEY,E,P` | E/YES | Select/confirm |
| `KEY,.,P` | NO/decimal | Cancel/back |
| `KEY,L/O,P` | L/O | Temporary lockout |
| `KEY,H,P` | HOLD | Hold on channel |
| `KEY,S,P` | SCAN | Resume scanning |
| `KEY,1,P` ... `KEY,9,P` | Number keys | Quick key toggle |
| `KEY,F,P` | FUNCTION | Function modifier |
| `KEY,<,P` | Left arrow | Navigate left |
| `KEY,>,P` | Right arrow | Navigate right |
| `KEY,^,P` | Up arrow | Navigate up |
| `KEY,V,P` | Down arrow | Navigate down |

## Quick Keys (SQK/DQK/FQK)

Enable or disable scanning groups remotely:

```
SQK,01,1    — Enable System Quick Key 1
SQK,01,0    — Disable System Quick Key 1
DQK,05,1    — Enable Department Quick Key 5
DQK,05,0    — Disable Department Quick Key 5
FQK,01,1    — Enable Favorites List Quick Key 1
```

## Useful Patterns

### Force Resume Scanning
```
KEY,S,P     — Equivalent to pressing SCAN button
```

### Temporary Lockout (Skip Channel)
```
KEY,L/O,P   — Lock out current channel until power cycle
```

### Check If Receiving
```
GLG         — Check squelch field (1=active transmission)
```

## Notes

- Commands must end with `\r` (0x0D)
- Response ends with `\r`
- Baud rate: 115200, 8N1
- Some commands only work when scanner is in specific modes
- The BCD436HP uses the same command set as BCD396XT (mostly compatible)
- STS response contains control characters from the LCD protocol (filter these)

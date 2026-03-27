//! Terminal UI for the local CUDA vanity miner.
//! With CLI args, execs the miner binary (for scripts): `uniq-tui -- --prefix …`

mod miner;

use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

enum Screen {
    Menu { selected: usize },
    Pattern {
        is_prefix: bool,
        input: String,
        case_sensitive: bool,
    },
    Mining {
        lines: VecDeque<String>,
        child: std::process::Child,
        log_rx: mpsc::Receiver<String>,
        finished: bool,
        exit_code: Option<i32>,
    },
}

struct App {
    screen: Screen,
    status: String,
}

impl App {
    fn new() -> Self {
        Self {
            screen: Screen::Menu { selected: 0 },
            status: "↑↓/jk select · Enter · q quit".into(),
        }
    }
}

const MENU: &[&str] = &[
    "Address prefix",
    "Address suffix",
    "Quit",
];

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        miner::ensure_cuda_miner();
        std::process::exit(miner::run_passthrough(&args));
    }

    miner::ensure_cuda_miner();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(ratatui::backend::CrosstermBackend::new(stdout))?;

    let mut app = App::new();
    let tick = Duration::from_millis(80);
    let res = run_loop(&mut terminal, &mut app, tick);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn run_loop(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut App,
    tick: Duration,
) -> io::Result<()> {
    loop {
        if let Screen::Mining {
            ref mut child,
            ref log_rx,
            ref mut lines,
            ref mut finished,
            ref mut exit_code,
            ..
        } = app.screen
        {
            while let Ok(line) = log_rx.try_recv() {
                if lines.len() >= 500 {
                    lines.pop_front();
                }
                lines.push_back(line);
            }
            if !*finished {
                match child.try_wait() {
                    Ok(Some(st)) => {
                        *finished = true;
                        *exit_code = st.code();
                        app.status = format!(
                            "Exit code {}. Enter — menu · q — quit",
                            exit_code.unwrap_or(-1)
                        );
                    }
                    Ok(None) => {}
                    Err(e) => {
                        *finished = true;
                        app.status = format!("Error: {e}");
                    }
                }
            }
        }

        terminal.draw(|f| ui(f, app))?;

        if event::poll(tick)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                if handle_key(app, key.code)? {
                    return Ok(());
                }
            }
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) -> io::Result<bool> {
    match &mut app.screen {
        Screen::Menu { selected } => match code {
            KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
            KeyCode::Down | KeyCode::Char('j') => {
                *selected = (*selected + 1).min(MENU.len().saturating_sub(1));
            }
            KeyCode::Up | KeyCode::Char('k') => {
                *selected = selected.saturating_sub(1);
            }
            KeyCode::Enter => match *selected {
                0 => {
                    app.screen = Screen::Pattern {
                        is_prefix: true,
                        input: String::new(),
                        case_sensitive: false,
                    };
                    app.status =
                        "Tab — case · Enter — start · Esc — back".into();
                }
                1 => {
                    app.screen = Screen::Pattern {
                        is_prefix: false,
                        input: String::new(),
                        case_sensitive: false,
                    };
                    app.status =
                        "Tab — case · Enter — start · Esc — back".into();
                }
                2 => return Ok(true),
                _ => {}
            },
            _ => {}
        },
        Screen::Pattern {
            is_prefix,
            input,
            case_sensitive,
        } => match code {
            KeyCode::Esc => {
                app.screen = Screen::Menu { selected: 0 };
                app.status = "↑↓/jk select · Enter · q quit".into();
            }
            KeyCode::Tab => {
                *case_sensitive = !*case_sensitive;
            }
            KeyCode::Enter => {
                if input.is_empty() {
                    app.status = "Enter a non-empty pattern".into();
                } else {
                    let is_p = *is_prefix;
                    let pat = input.clone();
                    let cs = *case_sensitive;
                    match spawn_miner(is_p, &pat, cs) {
                        Ok((child, rx, cmd_line)) => {
                            let mut lines = VecDeque::new();
                            lines.push_back(format!("$ {cmd_line}"));
                            app.screen = Screen::Mining {
                                lines,
                                child,
                                log_rx: rx,
                                finished: false,
                                exit_code: None,
                            };
                            app.status = "q — stop, back to menu".into();
                        }
                        Err(e) => {
                            app.status = format!("Spawn failed: {e}");
                        }
                    }
                }
            }
            KeyCode::Backspace => {
                input.pop();
            }
            KeyCode::Char(c) => {
                if !c.is_control() {
                    input.push(c);
                }
            }
            _ => {}
        },
        Screen::Mining {
            child,
            finished,
            ..
        } => match code {
            KeyCode::Char('q') => {
                let _ = child.kill();
                app.screen = Screen::Menu { selected: 0 };
                app.status = "↑↓/jk select · Enter · q quit".into();
            }
            KeyCode::Enter if *finished => {
                app.screen = Screen::Menu { selected: 0 };
                app.status = "↑↓/jk select · Enter · q quit".into();
            }
            _ => {}
        },
    }
    Ok(false)
}

fn spawn_miner(
    is_prefix: bool,
    pattern: &str,
    case_sensitive: bool,
) -> io::Result<(
    std::process::Child,
    mpsc::Receiver<String>,
    String,
)> {
    let bin = miner::cuda_bin();
    let ld = miner::ld_library_path_value();
    let cs = if case_sensitive { "true" } else { "false" };
    let cmd_line = format!(
        "{} {}{} --case-sensitive {cs} --stop-after-keys 1 --max-iterations 50000000 --attempts-per-execution 2000000",
        bin.display(),
        if is_prefix { "--prefix " } else { "--suffix " },
        pattern,
    );
    let mut cmd = Command::new(&bin);
    if is_prefix {
        cmd.arg("--prefix").arg(pattern);
    } else {
        cmd.arg("--suffix").arg(pattern);
    }
    cmd.arg("--case-sensitive")
        .arg(cs)
        .arg("--stop-after-keys")
        .arg("1")
        .arg("--max-iterations")
        .arg("50000000")
        .arg("--attempts-per-execution")
        .arg("2000000")
        .env("LD_LIBRARY_PATH", ld)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let out = child.stdout.take().expect("stdout");
    let err = child.stderr.take().expect("stderr");
    let (tx, rx) = mpsc::channel::<String>();
    let txo = tx.clone();
    thread::spawn(move || {
        let r = BufReader::new(out);
        for line in r.lines() {
            match line {
                Ok(l) => {
                    if txo.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    thread::spawn(move || {
        let r = BufReader::new(err);
        for line in r.lines() {
            match line {
                Ok(l) => {
                    if tx.send(format!("[stderr] {l}")).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok((child, rx, cmd_line))
}

fn ui(f: &mut Frame<'_>, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(2),
        ])
        .split(f.area());

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " uniq-tui ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  CUDA vanity miner"),
    ]))
    .block(Block::default().borders(Borders::BOTTOM));
    f.render_widget(title, chunks[0]);

    let main_block = Block::default().borders(Borders::ALL).title(" Miner ");

    match &mut app.screen {
        Screen::Menu { selected } => {
            let menu_lines: Vec<Line> = MENU
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if i == *selected {
                        Line::from(Span::styled(
                            format!("▶ {s}"),
                            Style::default().fg(Color::Cyan).bold(),
                        ))
                    } else {
                        Line::from(Span::styled(
                            format!("  {s}"),
                            Style::default().fg(Color::White),
                        ))
                    }
                })
                .collect();
            let p = Paragraph::new(menu_lines)
                .block(main_block)
                .wrap(Wrap { trim: true });
            f.render_widget(p, chunks[1]);
        }
        Screen::Pattern {
            is_prefix,
            input,
            case_sensitive,
        } => {
            let mode = if *is_prefix {
                "Prefix"
            } else {
                "Suffix"
            };
            let reg = if *case_sensitive {
                "case-sensitive"
            } else {
                "case-insensitive (faster)"
            };
            let text = vec![
                Line::from(""),
                Line::from(vec![
                    Span::styled("Mode: ", Style::default().fg(Color::Yellow)),
                    Span::raw(mode),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Pattern: ", Style::default().fg(Color::Green)),
                    Span::raw(input.as_str()),
                    Span::styled("▏", Style::default().fg(Color::Cyan)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Case (Tab): ", Style::default().fg(Color::Yellow)),
                    Span::raw(reg),
                ]),
                Line::from(""),
                Line::from(
                    Span::styled(
                        "Enter — start · Esc — menu",
                        Style::default().fg(Color::DarkGray),
                    ),
                ),
            ];
            let p = Paragraph::new(text).block(main_block).wrap(Wrap { trim: true });
            f.render_widget(p, chunks[1]);
        }
        Screen::Mining {
            lines,
            finished,
            exit_code,
            ..
        } => {
            let title = if *finished {
                format!(" Log (exit {}) ", exit_code.unwrap_or(-1))
            } else {
                " Log ".into()
            };
            let log_text: Vec<Line> = lines.iter().map(|s| Line::from(s.as_str())).collect();
            let p = Paragraph::new(log_text)
                .block(Block::default().borders(Borders::ALL).title(title))
                .wrap(Wrap { trim: false });
            f.render_widget(p, chunks[1]);
        }
    }

    let status = Paragraph::new(app.status.as_str()).style(Style::default().fg(Color::Gray));
    f.render_widget(status, chunks[2]);
}

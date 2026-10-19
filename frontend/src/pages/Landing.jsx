
import { Link } from "react-router-dom";
import { useEffect, useRef } from "react";
import screenshot from "../assets/screenshot.png";
import driveLogo from "../assets/vite.svg";
import westImg from "../assets/west.jpeg";
import northImg from "../assets/north.jpeg";
import farm1 from "../assets/farm1.jpg";
import farm2 from "../assets/farm2.webp";
import driveLabLogo from "../assets/LOGO+3.webp";

const GALLERY_ITEMS = [
  {
    src: westImg,
    caption: "",
  },
  {
    src: northImg,
    caption: "",
  },
  {
    src: farm1,
    caption: "",
  },
  {
    src: farm2,
    caption: "",
  }
];

export default function Landing() {
  const navRef = useRef(null);

  useEffect(() => {
    const onScroll = () => {
      if (!navRef.current) return;
      navRef.current.classList.toggle("scrolled", window.scrollY > 20);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="landing-root">

      {/* ── Navigation ── */}
      <nav ref={navRef} className="landing-nav">
        <span className="landing-brand">Wild Bird Activity<em> Monitor</em></span>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <a href="https://www.drivelabresearch.com/" target="_blank" rel="noopener noreferrer">
            <img src={driveLabLogo} alt="DRIVE Lab" style={{ height: 30, width: "auto", display: "block" }} />
          </a>
          <Link to="/login" style={{ fontSize: 13, color: "var(--text-secondary)", textDecoration: "none", fontWeight: 500 }}>
            Sign in
          </Link>
          <Link to="/login" className="cta-primary" style={{ padding: "9px 18px", fontSize: 13 }}>
            Get access
          </Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="hero">
        {/* Left: copy */}
        <div style={{ animation: "fadeUp 0.7s ease both" }}>
          <div className="hero-eyebrow">Wild bird activity — automated monitoring</div>
          <h1 className="hero-h1">
            Know how bird activity<br />
            <em>changes over time</em>
          </h1>
          <p className="hero-p">
            The Wild Bird Activity Monitor was created to provide a consistent
             and repeatable measure of wild bird activity. Because bird activity 
             can vary substantially across days, times, and locations, occasional observations 
             may not capture meaningful patterns. The system uses repeated camera recordings to 
             calculate a Bird Activity Index based on automated detections per unit of video time, 
             allowing users to establish baseline activity, track changes over time, compare locations, 
             and evaluate patterns associated with environmental conditions or management practices.
          </p>
          <div className="hero-ctas">
            <Link to="/login" className="cta-primary">
              Process a recording
              <span style={{ fontSize: 16 }}>→</span>
            </Link>
            <Link to="/login" className="cta-secondary">
              Explore observations
            </Link>
          </div>

          {/* Tiny proof points */}
          <div style={{ display: "flex", gap: 24, marginTop: 36, flexWrap: "wrap" }}>
            {[
              ["Track activity", "see changes over time"],
              ["Compare sites", "monitor multiple locations"],
              ["Explore your data", "summary plots and raw dataset"],
            ].map(([top, bot]) => (
              <div key={top} style={{ borderLeft: "2px solid var(--sage-mid)", paddingLeft: 12 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "var(--charcoal)" }}>{top}</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1 }}>{bot}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: screenshot */}
        <div className="hero-visual">
          <img src={screenshot} alt="Bird detection output" style={{ width: "100%", borderRadius: 8, display: "block" }} />
          {/* Corner decoration */}
          <div style={{
            position: "absolute", bottom: -14, right: -14,
            width: 100, height: 100,
            border: "1px solid var(--sage-mid)",
            borderRadius: 4, zIndex: -1, opacity: 0.4,
          }} />
        </div>
      </section>

      {/* ── Gallery ── */}
      <section style={{ background: "var(--charcoal)", padding: "32px 0" }}>
        <div className="gallery-strip">
          {GALLERY_ITEMS.map(({ src, caption }) => (
            <div key={caption} className="gallery-item">
              <img src={src} alt={caption}
                onError={e => { e.currentTarget.style.opacity = 0; }}
              />
              <div className="gallery-item-cap">{caption}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Method ── */}
      <section className="landing-section">
        <div className="section-title">How it works</div>
        <p className="section-body">
          To use the Wild Bird Activity Monitor, collect repeated skyline recordings from the same location. 
          Upload the recordings to the app, which will automatically process the videos and generate summary plots 
          of time-standardized detections and bird activity patterns. 
          You can also download the underlying raw dataset for further analysis. 
          Over time, repeated recordings will begin to reveal consistent patterns in bird activity and show how activity
          changes across days and locations.
          We recommend starting with at least 30 minutes of video per day for one week, and you can monitor up to
          five locations at a time. Longer recordings and more frequent sampling provide more information and may 
          allow a stable Bird Activity Index to be reached sooner. However, locations with highly variable bird activity
          may require additional days of recording to adequately capture normal day-to-day variation.
        </p>

        <div className="steps">
          {[
            { n: "01", title: "Set up cameras at key locations", desc: "Place a camera with a clear view of the skyline at each location you want to monitor. Keep the camera position consistent across recordings so activity can be compared over time. Up to five locations can be monitored separately." },
            { n: "02", title: "Record and upload footage", desc: "Collect repeated video recordings from each location and upload them to the app. Enter the recording date and start time so detections can be standardized by video duration and evaluated over time." },
            { n: "03", title: "Automated detection runs", desc: "The system automatically detects and tracks flying birds throughout each recording. These detections are standardized by video time to generate a Bird Activity Index for each recording and location."},
            { n: "04", title: "Explore activity patterns", desc: "Review summary plots to see how bird activity changes over time and across locations. As repeated recordings accumulate, you can identify baseline activity, recurring patterns, and periods of increased or decreased activity. The underlying detection data can also be downloaded for further analysis. " },
          ].map(({ n, title, desc }) => (
            <div key={n} className="step">
              <div className="step-num">{n}</div>
              <div className="step-title">{title}</div>
              <div className="step-desc">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA band ── */}
      <section style={{
        background: "var(--forest-deep)",
        padding: "64px 40px",
        textAlign: "center",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 40, flexWrap: "wrap" }}>
          <a href="https://www.drivelabresearch.com/" target="_blank" rel="noopener noreferrer">
            <img src={driveLabLogo} alt="DRIVE Lab" style={{ height: 80, width: "auto", objectFit: "contain" }} />
          </a>
          <div>
            <div style={{
              fontFamily: "'DM Serif Display', Georgia, serif",
              fontSize: 30,
              color: "#fff",
              marginBottom: 14,
              letterSpacing: -0.5,
            }}>
              Let's get started!
            </div>
            <p style={{ color: "rgba(255,255,255,0.45)", fontSize: 14, marginBottom: 28 }}>
              Create an account, add your camera locations, and start turning field footage into actionable data.
            </p>
            <Link to="/login" className="cta-primary">
              Create an account →
            </Link>
          </div>
          
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="landing-footer">
        Wild Bird Activity Monitor · DRIVE Lab · <a href="https://www.drivelabresearch.com/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>drivelabresearch.com</a> · Michigan State University · College of Veterinary Medicine
      </footer>
    </div>
  );
}

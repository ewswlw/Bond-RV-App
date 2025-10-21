# Bond RV App - Documentation Index

**Last Updated**: October 21, 2025

---

## 📚 Documentation Structure

This folder contains all documentation for the Bond RV App data pipeline, organized by topic.

---

## 🚀 Setup

**Start here if you're setting up the pipeline for the first time.**

### [QUICKSTART.md](Setup/QUICKSTART.md)
- 5-minute quick start guide
- Installation instructions
- Basic usage examples
- First-time setup checklist

---

## 🔄 Workflows

**Step-by-step guides for common tasks.**

### [Dropbox-Workflow.md](Workflows/Dropbox-Workflow.md) ⭐ PRIMARY WORKFLOW
- Complete setup instructions for Dropbox + GitHub workflow
- Computer 1 initial setup
- Computer 2+ setup instructions
- Daily workflow for adding new data
- Pulling code updates
- Troubleshooting guide
- Quick reference commands (Mac/Linux/Windows)

---

## 🏗️ Architecture

**Technical documentation and design decisions.**

### [bond_pipeline_documentation.md](Architecture/bond_pipeline_documentation.md)
- Complete technical documentation
- Data discovery & analysis
- Q&A session with requirements
- Business logic decisions
- Implementation details
- Test results and validation
- Performance metrics
- Production recommendations

---

## 📖 Reference

**Reference materials and decision documents.**

### [DELIVERABLES.txt](Reference/DELIVERABLES.txt)
- Project deliverables summary
- Module descriptions
- Test results
- Performance metrics
- Next steps

### [Data-Distribution-Options.md](Reference/Data-Distribution-Options.md)
- Analysis of 5 different data distribution strategies
- Pros/cons of each approach
- Decision matrix
- Cost analysis
- Selected approach rationale

---

## 📋 Quick Navigation

### I want to...

| Task | Document |
|------|----------|
| Set up the pipeline for the first time | [Setup/QUICKSTART.md](Setup/QUICKSTART.md) |
| Use Dropbox to share data across computers | [Workflows/Dropbox-Workflow.md](Workflows/Dropbox-Workflow.md) |
| Understand how the pipeline works | [Architecture/bond_pipeline_documentation.md](Architecture/bond_pipeline_documentation.md) |
| See what was delivered | [Reference/DELIVERABLES.txt](Reference/DELIVERABLES.txt) |
| Compare data distribution options | [Reference/Data-Distribution-Options.md](Reference/Data-Distribution-Options.md) |

---

## 🗂️ Folder Organization

```
Documentation/
├── README.md                           # This file
│
├── Setup/                              # Getting started guides
│   └── QUICKSTART.md                   # 5-minute setup guide
│
├── Workflows/                          # Step-by-step procedures
│   └── Dropbox-Workflow.md             # Dropbox + GitHub workflow
│
├── Architecture/                       # Technical design docs
│   └── bond_pipeline_documentation.md  # Complete technical docs
│
└── Reference/                          # Reference materials
    ├── DELIVERABLES.txt                # Project summary
    └── Data-Distribution-Options.md    # Data strategy analysis
```

---

## 🎯 Recommended Reading Order

### For New Users:
1. **[Setup/QUICKSTART.md](Setup/QUICKSTART.md)** - Get up and running
2. **[Workflows/Dropbox-Workflow.md](Workflows/Dropbox-Workflow.md)** - Learn the daily workflow
3. **[Reference/DELIVERABLES.txt](Reference/DELIVERABLES.txt)** - Understand what's included

### For Developers:
1. **[Architecture/bond_pipeline_documentation.md](Architecture/bond_pipeline_documentation.md)** - Understand the design
2. **[Reference/Data-Distribution-Options.md](Reference/Data-Distribution-Options.md)** - See why we chose Dropbox
3. **[Workflows/Dropbox-Workflow.md](Workflows/Dropbox-Workflow.md)** - Implement the workflow

### For Decision Makers:
1. **[Reference/DELIVERABLES.txt](Reference/DELIVERABLES.txt)** - See what was built
2. **[Reference/Data-Distribution-Options.md](Reference/Data-Distribution-Options.md)** - Understand data strategy
3. **[Architecture/bond_pipeline_documentation.md](Architecture/bond_pipeline_documentation.md)** - Review technical decisions

---

## 📝 Document Maintenance

### Adding New Documentation

When adding new documentation:

1. **Choose the right folder:**
   - `Setup/` - Installation, configuration, getting started
   - `Workflows/` - Step-by-step procedures, how-to guides
   - `Architecture/` - Technical design, system architecture
   - `Reference/` - Decision documents, comparisons, summaries

2. **Use consistent naming:**
   - Use kebab-case: `My-Document.md`
   - Be descriptive: `Dropbox-Workflow.md` not `Workflow.md`

3. **Update this README:**
   - Add link in appropriate section
   - Update quick navigation table
   - Update folder organization diagram

4. **Include metadata:**
   ```markdown
   # Document Title
   
   **Document Version**: 1.0
   **Date**: YYYY-MM-DD
   **Purpose**: Brief description
   ```

### Document Versioning

- Update version number when making significant changes
- Update "Last Updated" date in this README
- Consider adding changelog section for major documents

---

## 🔍 Search Tips

### Finding Information

Use your editor's search (Ctrl+F / Cmd+F) or grep:

```bash
# Search all documentation
grep -r "CUSIP" Documentation/

# Search specific folder
grep -r "Dropbox" Documentation/Workflows/

# Case-insensitive search
grep -ri "error" Documentation/
```

### Common Keywords

- **Setup**: installation, dependencies, requirements
- **Workflow**: procedure, steps, how-to
- **Troubleshooting**: error, issue, problem, solution
- **Data**: CUSIP, parquet, Excel, universe, historical
- **Commands**: pipeline.py, append, override

---

## 📞 Support

If you can't find what you're looking for:

1. Check the [Quick Navigation](#quick-navigation) table above
2. Search documentation using keywords
3. Review troubleshooting sections in workflow docs
4. Check logs in `bond_data/logs/`

---

## 📄 License

All documentation is proprietary and confidential.

---

**Maintained by**: Bond RV App Team  
**Repository**: https://github.com/ewswlw/Bond-RV-App


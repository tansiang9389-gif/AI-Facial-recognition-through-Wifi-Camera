"""
Face Manager — View, rename, and delete persons in the database.

Usage:
    python name_faces.py              Show all persons, rename interactively
    python name_faces.py --list       Just list all persons
    python name_faces.py --rename P0001 "John Smith"
    python name_faces.py --delete P0003
    python name_faces.py --show P0001  Show the face image
"""

import json
import os
import sys
import argparse

FACE_DB_FILE = "face_database.json"
KNOWN_FACES_DIR = "known_faces"


def load_db():
    if not os.path.exists(FACE_DB_FILE):
        print("No face database found. Run the main system first.")
        sys.exit(1)
    with open(FACE_DB_FILE, "r") as f:
        return json.load(f)


def save_db(data):
    with open(FACE_DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


def list_persons(data):
    persons = data.get("persons", {})
    if not persons:
        print("Database is empty.")
        return

    print(f"\n{'='*65}")
    print(f"  Face Database — {len(persons)} person(s)")
    print(f"{'='*65}")
    print(f"  {'ID':<8} {'Name':<22} {'First Seen':<20} {'Image'}")
    print(f"  {'─'*8} {'─'*22} {'─'*20} {'─'*15}")

    for pid, info in sorted(persons.items()):
        img_exists = "✓" if os.path.exists(os.path.join(KNOWN_FACES_DIR, info["image"])) else "✗"
        print(f"  {pid:<8} {info['name']:<22} {info['first_seen']:<20} {img_exists} {info['image']}")

    print(f"{'='*65}\n")


def interactive_rename(data):
    persons = data.get("persons", {})
    unnamed = {pid: info for pid, info in persons.items() if info["name"].startswith("Person ")}

    if not unnamed:
        print("All persons are already named!")
        return

    print(f"\n{len(unnamed)} unnamed person(s). Opening images for identification...\n")

    try:
        import cv2
        can_show = True
    except ImportError:
        can_show = False

    for pid, info in sorted(unnamed.items()):
        img_path = os.path.join(KNOWN_FACES_DIR, info["image"])

        if can_show and os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imshow(f"{pid} - {info['name']}", img)
                cv2.waitKey(500)

        print(f"  {pid}: {info['name']}")
        print(f"    Image: {info['image']}")
        print(f"    First seen: {info['first_seen']}")
        new_name = input(f"    Enter name (or ENTER to skip, 'd' to delete): ").strip()

        if can_show:
            cv2.destroyAllWindows()

        if new_name.lower() == 'd':
            # Delete this person
            del persons[pid]
            if os.path.exists(img_path):
                os.remove(img_path)
            print(f"    Deleted {pid}\n")
        elif new_name:
            persons[pid]["name"] = new_name
            print(f"    Renamed -> {new_name}\n")
        else:
            print(f"    Skipped\n")

    save_db(data)
    print("Database saved.")


def rename_person(data, pid, new_name):
    persons = data.get("persons", {})
    if pid not in persons:
        print(f"Person {pid} not found.")
        return
    old = persons[pid]["name"]
    persons[pid]["name"] = new_name
    save_db(data)
    print(f"Renamed: {old} -> {new_name}")


def delete_person(data, pid):
    persons = data.get("persons", {})
    if pid not in persons:
        print(f"Person {pid} not found.")
        return
    info = persons[pid]
    img_path = os.path.join(KNOWN_FACES_DIR, info["image"])
    del persons[pid]
    if os.path.exists(img_path):
        os.remove(img_path)
    save_db(data)
    print(f"Deleted: {pid} ({info['name']})")


def show_person(pid):
    data = load_db()
    persons = data.get("persons", {})
    if pid not in persons:
        print(f"Person {pid} not found.")
        return
    info = persons[pid]
    img_path = os.path.join(KNOWN_FACES_DIR, info["image"])
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    try:
        import cv2
        img = cv2.imread(img_path)
        cv2.imshow(f"{pid} - {info['name']}", img)
        print(f"Showing {pid}: {info['name']}. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ImportError:
        print(f"OpenCV not available. Image is at: {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the face database")
    parser.add_argument("--list", action="store_true", help="List all persons")
    parser.add_argument("--rename", nargs=2, metavar=("PID", "NAME"), help="Rename a person")
    parser.add_argument("--delete", metavar="PID", help="Delete a person")
    parser.add_argument("--show", metavar="PID", help="Show a person's face image")
    args = parser.parse_args()

    data = load_db()

    if args.list:
        list_persons(data)
    elif args.rename:
        rename_person(data, args.rename[0], args.rename[1])
    elif args.delete:
        delete_person(data, args.delete)
    elif args.show:
        show_person(args.show)
    else:
        list_persons(data)
        interactive_rename(data)

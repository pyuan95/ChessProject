#pragma once
#include <iostream>

template <class T>
class PriorityQueue {
private:
	vector<pair<T, float>> arr;
	uint16_t s;

	void bubble_up(int i);
	void bubble_down(int i);
	int largest_child(int i);
	int parent(int i);
public:
	PriorityQueue() : arr(), s(0) {}

	/*
	Returns the number of elements in the queue
	*/
	inline size_t size() { return s; }

	/*
	Updates the root with the given priority, and adjusts the priority queue accordingly.
	*/
	void update_root(float priority);

	/*
	Pushes an element onto the priority queue with the given priority.
	*/
	void push(T e, float priority);

	/*
	Returns the element with the greatest priority.
	*/
	T peek();

	inline void print() { for (int i = 0; i < s; i++) std::cout << arr[i].second << " "; std::cout << "\n"; }
};

template <class T>
int PriorityQueue<T>::largest_child(int i)
{
	int c1 = 2 * i + 1;
	int c2 = c1 + 1;
	if (c2 < s && arr[c2].second > arr[c1].second)
		return c2;
	return c1; // either both in bounds and c1 is greater, or c2 is out of bounds. Either way return c1.
}

template <class T>
int PriorityQueue<T>::parent(int i)
{
	return i <= 0 ? -1 : (i - 1) / 2;
}

template <class T>
void PriorityQueue<T>::bubble_up(int i)
{
	int par = parent(i);
	while (par >= 0 && arr[par].second < arr[i].second)
	{
		auto temp = arr[i];
		arr[i] = arr[par];
		arr[par] = temp;
		i = par;
		par = parent(i);
	}
}

template <class T>
void PriorityQueue<T>::bubble_down(int i)
{
	int lc = largest_child(i);
	while (lc < s && arr[i].second < arr[lc].second)
	{
		auto temp = arr[i];
		arr[i] = arr[lc];
		arr[lc] = temp;
		i = lc;
		lc = largest_child(i);
	}
}

template<class T>
inline void PriorityQueue<T>::update_root(float priority)
{
	arr[0].second = priority;
	bubble_down(0);
}

template<class T>
inline void PriorityQueue<T>::push(T e, float priority)
{
	s++;
	arr.push_back(pair<T, float>(e, priority));
	bubble_up(s - 1);
}

template<class T>
inline T PriorityQueue<T>::peek()
{
	return arr[0].first;
}